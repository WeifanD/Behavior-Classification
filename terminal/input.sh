#!/bin/sh
#set -e
# config operate_date
DATE=$(date +%Y-%m-%d)
YESTERDAY=`date -d "$DATE -1 day" +%Y-%m-%d`


echo "YESTERDAY------------$YESTERDAY"

#当日注册的用户
/usr/lib/spark-current/bin/beeline -u 'jdbc:hive2://172.16.53.159:10000' -n ac -p ac123 --outputformat=csv2 --incremental=true -e "set mapreduce.job.queuename=root.develop.adhoc.ac;

set mapreduce.job.queuename=root.develop.adhoc.ac;
select teacher_id, count(distinct member_id) as tds 
from gobblin.qukan_p_member_info 
where day = '${YESTERDAY}'
and teacher_id > 100000000
and to_date(update_time) = '${YESTERDAY}'
group by teacher_id
having tds >= 8
" > input.csv

echo "Input file ready."