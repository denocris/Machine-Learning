
* First Cleaning on Text

cat yelptest.json | sed -e 's/"text":/\'$'\n"text":/g' | sed 's/"type":/\'$'\n"type:/g' | sed '/^"type/d' | sed '/"votes"/d' | sed 's/"text": //' | sed -e 's/\\n//g' | tr -d , | tr -d \" | sed -e 's/^/ /' > newfile.file

* Select stars

cat oldfile.file | sed -e 's/"stars":/\'$'\n"stars":/g' | sed 's/"date":/\'$'\n"date:/g'| sed '/^"date/d' | sed '/"votes"/d'| sed 's/"stars": //' | tr -d , > newfile.file

* Remove \n in text

sed -e 's/\\n//g' oldfile.file > newfile.file

* To visualize text line by line
nano -w file.txt
