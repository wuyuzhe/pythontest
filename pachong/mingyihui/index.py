#coding=utf-8
import urllib
import urllib.request
import time
from bs4 import BeautifulSoup

ENCODING = 'utf-8'

def url_request(url):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    req = urllib.request.Request(url=url,headers=headers)
    result = urllib.request.urlopen(req).read()
    return result

#print(result.decode())

def parse(html):
    soup = BeautifulSoup(html,"html5lib")
    h1 = soup.find('h1' , attrs={'class':'hospital_title'})
    hospital_name = h1.get_text()

    line = []
    line.append(hospital_name)
    p_b = soup.find_all('p' , attrs={'class':'overview_p'})
    for p in p_b:
        string = p.get_text()
        string = string.replace('\n','')
        string = string.replace(',','&')
        string = string.strip()
        line.append(string)
    line.append('\n')
    html = ','.join(line)
    return html

if __name__ == '__main__':
    f = open('a.txt','a+',encoding='utf-8')
    for i in range(12502,14000):
        try:
            url = 'http://www.mingyihui.net/hospital_%i.html' %(i)
            result = url_request(url)
            line = parse(result.decode())
            f.write(str(i)+',')
            f.write(line)
        except Exception as e:
             print('Error:',e)
             continue
        if i%20 == 0:#进度输出
            print(line)
            time.sleep(1)
    f.close()
