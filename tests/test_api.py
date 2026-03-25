import pytest
import sys
import os
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from main import app
from services import conversation_service
from unittest.mock import patch, MagicMock

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_conversations():
    """Clean up conversation files before and after each test."""
    conversations = conversation_service.get_all_conversations()
    for c in conversations:
        conversation_service.delete_conversation(c.id)
    yield
    conversations = conversation_service.get_all_conversations()
    for c in conversations:
        conversation_service.delete_conversation(c.id)


@pytest.fixture
def mock_openai():
    """Mock OpenAI API calls."""
    mock_client_instance = MagicMock()

    # Mock title generation - returns title response
    mock_title_response = MagicMock()
    mock_title_response.choices = [MagicMock(message=MagicMock(content='"Test Conversation"'))]

    # Mock chat response - returns AI response
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [MagicMock(message=MagicMock(content="This is a test response."))]

    # Return title for first call, chat response for all subsequent calls
    mock_client_instance.chat.completions.create.side_effect = [
        mock_title_response,  # Title generation (first chat only)
        mock_chat_response,  # First chat
        mock_chat_response,  # Second chat
    ]

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
         patch("services.openai_client.create_openai_client", return_value=mock_client_instance):
        yield mock_client_instance


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestConversationEndpoints:
    def test_create_conversation(self):
        response = client.post("/conversations")
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["title"] == "New Chat"

    def test_list_conversations_empty(self):
        response = client.get("/conversations")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_conversations_returns_all(self, mock_openai):
        # Create multiple conversations
        for _ in range(3):
            client.post("/conversations")

        response = client.get("/conversations")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_list_conversations_ordered_by_updated_at(self, mock_openai):
        # Create conversations with a delay
        id1 = client.post("/conversations").json()["id"]
        id2 = client.post("/conversations").json()["id"]

        # Update first conversation
        client.patch(f"/conversations/{id1}/title", json={"title": "Updated"})

        response = client.get("/conversations")
        data = response.json()
        # Updated conversation should be first
        assert data[0]["id"] == id1
        assert data[0]["title"] == "Updated"

    def test_get_conversation_not_found(self):
        response = client.get("/conversations/nonexistent-id")
        assert response.status_code == 404

    def test_get_conversation_success(self, mock_openai):
        created = client.post("/conversations").json()
        response = client.get(f"/conversations/{created['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == created["id"]
        assert data["title"] == "New Chat"
        assert data["messages"] == []

    def test_rename_conversation(self, mock_openai):
        created = client.post("/conversations").json()
        response = client.patch(
            f"/conversations/{created['id']}/title",
            json={"title": "My Custom Title"}
        )
        assert response.status_code == 200
        assert response.json()["title"] == "My Custom Title"

    def test_rename_conversation_not_found(self):
        response = client.patch(
            "/conversations/nonexistent-id/title",
            json={"title": "Test"}
        )
        assert response.status_code == 404

    def test_delete_conversation(self, mock_openai):
        created = client.post("/conversations").json()
        response = client.delete(f"/conversations/{created['id']}")
        assert response.status_code == 200

        # Verify deleted
        response = client.get(f"/conversations/{created['id']}")
        assert response.status_code == 404

    def test_delete_conversation_not_found(self):
        response = client.delete("/conversations/nonexistent-id")
        assert response.status_code == 404


class TestChatEndpoint:
    def test_chat_without_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            response = client.post("/chat", json={
                "conversation_id": "test",
                "message": "Hello"
            })
            assert response.status_code == 500
            assert "API key not configured" in response.json()["detail"]

    def test_chat_empty_message(self, mock_openai):
        conv = client.post("/conversations").json()
        response = client.post("/chat", json={
            "conversation_id": conv["id"],
            "message": ""
        })
        assert response.status_code == 400

    def test_chat_conversation_not_found(self, mock_openai):
        response = client.post("/chat", json={
            "conversation_id": "nonexistent-id",
            "message": "Hello"
        })
        assert response.status_code == 404

    def test_chat_adds_messages(self, mock_openai):
        conv = client.post("/conversations").json()
        response = client.post("/chat", json={
            "conversation_id": conv["id"],
            "message": "Hello AI!"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == conv["id"]
        assert data["content"] == "This is a test response."
        assert data["title"] != "New Chat"  # Title was auto-generated

        # Verify messages in conversation
        conv_response = client.get(f"/conversations/{conv['id']}")
        messages = conv_response.json()["messages"]
        assert len(messages) == 2  # User message + AI response
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello AI!"
        assert messages[1]["role"] == "assistant"

    def test_chat_maintains_history(self, mock_openai):
        conv = client.post("/conversations").json()

        # First exchange
        client.post("/chat", json={
            "conversation_id": conv["id"],
            "message": "First message"
        })

        # Second exchange
        client.post("/chat", json={
            "conversation_id": conv["id"],
            "message": "Second message"
        })

        # Verify both messages in history
        conv_response = client.get(f"/conversations/{conv['id']}")
        messages = conv_response.json()["messages"]
        assert len(messages) == 4  # 2 user + 2 AI


class TestTitleGeneration:
    def test_title_not_generated_for_empty_first_message(self, mock_openai):
        with patch("services.title_generator.generate_title") as mock_gen:
            mock_gen.return_value = "Should Not Be Called"
            conv = client.post("/conversations").json()

            # Send whitespace-only message
            client.post("/chat", json={
                "conversation_id": conv["id"],
                "message": "   "
            })

            # Title should remain New Chat since message is essentially empty
            mock_gen.assert_not_called()


class TestConversationService:
    def test_create_and_retrieve_conversation(self):
        conversation = conversation_service.create_conversation()
        assert conversation.title == "New Chat"
        assert len(conversation.messages) == 0

        retrieved = conversation_service.get_conversation(conversation.id)
        assert retrieved is not None
        assert retrieved.id == conversation.id

    def test_add_message_to_conversation(self):
        conversation = conversation_service.create_conversation()
        result = conversation_service.add_message(
            conversation.id, "user", "Test message"
        )
        assert result is not None
        updated_conv, message = result
        assert message.role == "user"
        assert message.content == "Test message"
        assert len(updated_conv.messages) == 1

    def test_update_title(self):
        conversation = conversation_service.create_conversation()
        updated = conversation_service.update_title(conversation.id, "My Title")
        assert updated is not None
        assert updated.title == "My Title"

    def test_delete_conversation(self):
        conversation = conversation_service.create_conversation()
        assert conversation_service.delete_conversation(conversation.id) is True
        assert conversation_service.get_conversation(conversation.id) is None

    def test_get_all_conversations(self):
        conversation_service.create_conversation()
        conversation_service.create_conversation()
        all_conv = conversation_service.get_all_conversations()
        assert len(all_conv) == 2
