echo $(ps aux | grep $1 | grep hefb | grep -v grep | grep -v kill | awk '{print $2}')
kill -9 $(ps aux | grep $1 | grep hefb | grep -v grep | grep -v kill | awk '{print $2}')
