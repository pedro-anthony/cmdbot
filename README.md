# cmdBOT

A Discord bot powered by the Google Gemini API. It's designed to be a helpful and entertaining chat companion, featuring a persistent memory (cache), a knowledge base, and dynamic status updates based on conversation history.

## Features

*   **Google Gemini Integration:** Utilizes the Gemini API for intelligent and context-aware responses.
*   **Conversation Cache:** Maintains a history of conversations in each channel to provide better contextual answers. The cache is stored in `message_cache.pkl`.
*   **Knowledge Base:** Uses a `knowledge.json` file to store and retrieve information about users, facts, and concepts, making interactions more personalized.
*   **Dynamic Presence:** The bot's status on Discord dynamically updates based on the ongoing conversations, showing what it's "thinking" or "doing".
*   **Image Understanding:** Can process and understand images sent in the chat.
*   **Asynchronous:** Built with `asyncio` and `discord.py` for efficient, non-blocking operation.

## Setup

### Prerequisites

*   Python 3.11 or higher.
*   [Poetry](https://python-poetry.org/) for dependency management.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd cmdbot
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```

3.  **Create a `.env` file:**
    Create a file named `.env` in the root directory of the project and add your API keys:
    ```
    DISCORD_TOKEN=your_discord_bot_token
    GEMINI_API_KEY=your_gemini_api_key
    ```

## Usage

To run the bot, execute the `main.py` script:

```bash
python main.py
```

Once the bot is running and connected to your Discord server, you can interact with it by:

*   **Mentioning it:** `@cmdBOT`
*   **Using its name:** `cmdbot`

The bot will then process the message and respond in the channel.