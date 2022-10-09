import urllib
from urllib.request import urlopen


def notifyDimChig(text):
    #sends a TELEGRAM message to me when training is finished!
    text = text.replace(" ","+")
    bot_id = "xxxxxxx:yyyyyyyyyyyyyyyyyyyyyyyyy"
    userd_id = 1111111111
    url = f"https://api.telegram.org/bot{bot_id}/sendMessage?chat_id={userd_id}&parse_mode=markdown&text=" + text
    urlopen(url)