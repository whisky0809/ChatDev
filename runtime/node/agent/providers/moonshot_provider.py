"""Moonshot AI provider implementation for Kimi models."""

from typing import Any, Dict, List, Optional

from openai import OpenAI

from entity.messages import (
    Message,
    MessageRole,
    ToolCallPayload,
)
from entity.tool_spec import ToolSpec
from runtime.node.agent import ModelProvider
from runtime.node.agent import ModelResponse
from utils.token_tracker import TokenUsage


class MoonshotProvider(ModelProvider):
    """Moonshot AI provider implementation for Kimi models."""

    DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"

    def create_client(self):
        """
        Create and return the OpenAI client configured for Moonshot API.

        Returns:
            OpenAI client instance with Moonshot base URL
        """
        base_url = self.base_url or self.DEFAULT_BASE_URL
        return OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )

    def call_model(
        self,
        client: OpenAI,
        conversation: List[Message],
        timeline: List[Any],
        tool_specs: Optional[List[ToolSpec]] = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Call the Moonshot model with the given messages and parameters.

        Uses Chat Completions API since Moonshot doesn't support OpenAI's Responses API.
        """
        request_payload = self._build_chat_payload(conversation, tool_specs, kwargs)

        # Extract timeout for the API call (default 5 minutes)
        timeout = request_payload.pop("timeout", 300)

        response = client.chat.completions.create(**request_payload, timeout=timeout)

        self._track_token_usage(response)
        self._append_chat_response_output(timeline, response)
        message = self._deserialize_chat_response(response)

        return ModelResponse(message=message, raw_response=response)

    def extract_token_usage(self, response: Any) -> TokenUsage:
        """
        Extract token usage from the Moonshot API response.

        Args:
            response: Moonshot API response (OpenAI-compatible format)

        Returns:
            TokenUsage instance with token counts
        """
        usage = self._get_attr(response, "usage")
        if not usage:
            return TokenUsage()

        prompt_tokens = self._get_attr(usage, "prompt_tokens")
        completion_tokens = self._get_attr(usage, "completion_tokens")
        total_tokens = self._get_attr(usage, "total_tokens")

        if total_tokens is None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        metadata = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or 0,
        }

        return TokenUsage(
            input_tokens=prompt_tokens or 0,
            output_tokens=completion_tokens or 0,
            total_tokens=total_tokens or 0,
            metadata=metadata,
        )

    def _track_token_usage(self, response: Any) -> None:
        """Record token usage if a tracker is attached to the config."""
        token_tracker = getattr(self.config, "token_tracker", None)
        if not token_tracker:
            return

        usage = self.extract_token_usage(response)
        if usage.input_tokens == 0 and usage.output_tokens == 0 and not usage.metadata:
            return

        node_id = getattr(self.config, "node_id", "ALL")
        usage.node_id = node_id
        usage.model_name = self.model_name
        usage.workflow_id = token_tracker.workflow_id
        usage.provider = "moonshot"

        token_tracker.record_usage(node_id, self.model_name, usage, provider="moonshot")

    def _build_chat_payload(
        self,
        conversation: List[Message],
        tool_specs: Optional[List[ToolSpec]],
        raw_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct standard Chat Completions API payload for Moonshot."""
        params = dict(raw_params)
        max_output_tokens = params.pop("max_output_tokens", None)
        max_tokens = params.pop("max_tokens", None)
        if max_tokens is None and max_output_tokens is not None:
            max_tokens = max_output_tokens

        messages: List[Any] = []
        for item in conversation:
            serialized = self._serialize_message_for_chat(item)
            if serialized is not None:
                messages.append(serialized)

        if not messages:
            messages = [{"role": "user", "content": ""}]

        # Note: kimi-k2.5 only supports temperature=1.0
        # Using 1.0 as default for Moonshot to ensure compatibility
        temperature = params.pop("temperature", 1.0)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "timeout": params.pop("timeout", 300),  # 5 min default
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.params.get("max_tokens"):
            payload["max_tokens"] = self.params["max_tokens"]

        user_tools = params.pop("tools", None)
        merged_tools: List[Any] = []
        if isinstance(user_tools, list):
            merged_tools.extend(user_tools)

        if tool_specs:
            for spec in tool_specs:
                merged_tools.append({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters or {"type": "object", "properties": {}},
                    }
                })

        if merged_tools:
            payload["tools"] = merged_tools

        tool_choice = params.pop("tool_choice", None)
        if tool_choice is not None:
            # Moonshot doesn't support "required" tool_choice
            if tool_choice != "required":
                payload["tool_choice"] = tool_choice
        elif tool_specs:
            payload.setdefault("tool_choice", "auto")

        # Remove internal control params that shouldn't be sent to the API
        params.pop("protocol", None)

        payload.update(params)
        return payload

    def _serialize_message_for_chat(self, message: Message) -> Dict[str, Any]:
        """Convert internal Message to standard Chat Completions schema."""
        role_value = message.role.value
        blocks = message.blocks()
        if not blocks or message.role == MessageRole.TOOL:
            content = message.text_content()
        else:
            content = self._transform_blocks_for_chat(self._serialize_blocks(blocks, message.role))

        payload: Dict[str, Any] = {
            "role": role_value,
            "content": content,
        }
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [tc.to_openai_dict() for tc in message.tool_calls]
        # Include reasoning_content for models like Kimi K2.5 with thinking mode
        if message.metadata.get("reasoning_content"):
            payload["reasoning_content"] = message.metadata["reasoning_content"]
        return payload

    def _serialize_blocks(self, blocks: List[Any], role: MessageRole) -> List[Dict[str, Any]]:
        """Serialize message blocks for chat format."""
        from entity.messages import MessageBlock, MessageBlockType, AttachmentRef

        serialized: List[Dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, MessageBlock):
                if block.type is MessageBlockType.TEXT:
                    content_type = "output_text" if role is MessageRole.ASSISTANT else "input_text"
                    serialized.append({
                        "type": content_type,
                        "text": block.text or "",
                    })
                elif block.type in (MessageBlockType.IMAGE, MessageBlockType.AUDIO, MessageBlockType.VIDEO):
                    attachment = block.attachment
                    if attachment:
                        media_type = "output_image" if role is MessageRole.ASSISTANT else "input_image"
                        if block.type is MessageBlockType.AUDIO:
                            media_type = "output_audio" if role is MessageRole.ASSISTANT else "input_audio"
                        elif block.type is MessageBlockType.VIDEO:
                            media_type = "output_video" if role is MessageRole.ASSISTANT else "input_video"

                        payload: Dict[str, Any] = {"type": media_type}
                        if attachment.remote_file_id:
                            payload["file_id"] = attachment.remote_file_id
                        elif attachment.data_uri:
                            url_key = "image_url" if block.type is MessageBlockType.IMAGE else "url"
                            payload[url_key] = attachment.data_uri
                        serialized.append(payload)
                elif block.type is MessageBlockType.FILE:
                    # Convert file blocks to text for chat compatibility
                    attachment = block.attachment
                    if attachment:
                        filename = attachment.name or "file"
                        serialized.append({
                            "type": "input_text",
                            "text": f"[File: {filename}]",
                        })
        return serialized

    def _transform_blocks_for_chat(self, blocks: List[Dict[str, Any]]) -> Any:
        """Convert Responses block types to Chat block types."""
        transformed: List[Dict[str, Any]] = []
        for block in blocks:
            b_type = block.get("type", "")
            if b_type in ("input_text", "output_text"):
                transformed.append({"type": "text", "text": block.get("text", "")})
            elif b_type in ("input_image", "output_image"):
                transformed.append({"type": "image_url", "image_url": {"url": block.get("image_url", "")}})
            elif b_type in ("input_audio", "output_audio"):
                transformed.append({"type": "text", "text": "[Audio content]"})
            elif b_type in ("input_video", "output_video"):
                transformed.append({"type": "text", "text": "[Video content]"})
            else:
                transformed.append(block)

        # If only one text block, return as string for better compatibility
        if len(transformed) == 1 and transformed[0]["type"] == "text":
            return transformed[0]["text"]
        return transformed

    def _deserialize_chat_response(self, response: Any) -> Message:
        """Convert Chat Completions output to internal Message."""
        choices = self._get_attr(response, "choices") or []
        if not choices:
            return Message(role=MessageRole.ASSISTANT, content="")

        choice = choices[0]
        msg = self._get_attr(choice, "message")

        tool_calls: List[ToolCallPayload] = []
        tc_data = self._get_attr(msg, "tool_calls")
        if tc_data:
            for idx, tc in enumerate(tc_data):
                f_data = self._get_attr(tc, "function") or {}
                function_name = self._get_attr(f_data, "name") or ""
                arguments = self._get_attr(f_data, "arguments") or ""
                if not isinstance(arguments, str):
                    arguments = str(arguments)
                call_id = self._get_attr(tc, "id")
                if not call_id:
                    call_id = self._build_tool_call_id(function_name, arguments, fallback_prefix=f"tool_call_{idx}")
                tool_calls.append(ToolCallPayload(
                    id=call_id,
                    function_name=function_name,
                    arguments=arguments,
                    type="function"
                ))

        # Capture reasoning_content for models like Kimi K2.5 with thinking mode
        metadata = {}
        reasoning_content = self._get_attr(msg, "reasoning_content")
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content

        return Message(
            role=MessageRole.ASSISTANT,
            content=self._get_attr(msg, "content") or "",
            tool_calls=tool_calls,
            metadata=metadata
        )

    def _append_chat_response_output(self, timeline: List[Any], response: Any) -> None:
        """Add chat response to timeline, preserving tool_calls."""
        choices = self._get_attr(response, "choices") or []
        if not choices:
            return

        msg = choices[0].message
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or ""
        }

        # Preserve reasoning_content for models like Kimi K2.5 with thinking mode
        reasoning_content = self._get_attr(msg, "reasoning_content")
        if reasoning_content:
            assistant_msg["reasoning_content"] = reasoning_content

        tool_calls = self._get_attr(msg, "tool_calls")
        if tool_calls:
            assistant_msg["tool_calls"] = []
            for idx, tc in enumerate(tool_calls):
                function_name = tc.function.name
                arguments = tc.function.arguments or ""
                if not isinstance(arguments, str):
                    arguments = str(arguments)
                call_id = tc.id or self._build_tool_call_id(function_name, arguments, fallback_prefix=f"tool_call_{idx}")
                assistant_msg["tool_calls"].append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": arguments,
                    },
                })

        timeline.append(assistant_msg)

    def _get_attr(self, payload: Any, key: str) -> Any:
        """Safely get an attribute from an object or dict."""
        if hasattr(payload, key):
            return getattr(payload, key)
        if isinstance(payload, dict):
            return payload.get(key)
        return None

    def _build_tool_call_id(self, function_name: str, arguments: str, *, fallback_prefix: str = "tool_call") -> str:
        """Build a unique tool call ID."""
        import hashlib
        base = function_name or fallback_prefix
        payload = f"{base}:{arguments or ''}".encode("utf-8")
        digest = hashlib.md5(payload).hexdigest()[:8]
        return f"{base}_{digest}"
