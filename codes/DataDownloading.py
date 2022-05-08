import datetime
import os


def download_raw():
    today = datetime.datetime.today()
    if os.path.exists('./data/raw/DXYArea.csv'):
            os.remove('./data/raw/DXYArea.csv')

    print('\nPlease Wait...')

    os.system('curl -o ./data/raw/GlobalData.csv https://covid19.who.int/WHO-COVID-19-global-data.csv')

    print('Download Completed')

def download_news():
    pass

if __name__ == '__main__':
    download_raw()
    download_news()