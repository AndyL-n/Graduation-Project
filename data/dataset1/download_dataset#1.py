#! /usr/bin/env python
#
# Copyright (C) 2016, WS-DREAM, CUHK
# License: MIT

import urllib.request, urllib.error,urllib.parse
import re, os
import zipfile
import shutil


url = 'https://wsdream.github.io/dataset/wsdream_dataset1'
page = urllib.request.urlopen(url)


html = page.read()
html = html.decode('utf-8')   #python3
pattern = r'<a id="downloadlink" href="(.*?)"'
downloadlink = re.findall(pattern, html)[0]
file_name = url.split('/')[-1]
print(file_name)
page = urllib.request.urlopen(downloadlink)
#meta = page.info()
#file_size = int(meta.getheaders("Content-Length")[0])
file_size=page.getheader('Content-Length')
print("Downloading: %s (%s bytes)" % (file_name, file_size))
print("begin")
urllib.request.urlretrieve(downloadlink)
print("end")
print("Unzip data files...")
with zipfile.ZipFile(file_name, 'r') as z:
   for name in z.namelist():
      filename = os.path.basename(name)
      # skip directories
      if not filename:
         continue
      # copy file (taken from zipfile's extract)
      print( filename)
      source = z.open(name)
      target = open(filename, "wb")
      with source, target:
         shutil.copyfileobj(source, target)

os.remove(file_name)

print('==============================================')
print('Downloading data done!\n')

page.close()
