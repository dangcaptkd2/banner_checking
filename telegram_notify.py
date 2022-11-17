import logging
import requests

bot_token = ''
bot_chatID = ''


def telegram_bot_sendtext(bot_message):
    try:
        bot_message = bot_message.replace("_", "\\_")
        command = (
            "https://api.telegram.org/bot"
            + bot_token
            + "/sendMessage?chat_id="
            + bot_chatID
            + "&parse_mode=Markdown&text="
            + bot_message
        )
        
        response = requests.get(command)
        return response.json()
    except:
        return None


if __name__ == "__main__":
    telegram_bot_sendtext('test in server dev')
