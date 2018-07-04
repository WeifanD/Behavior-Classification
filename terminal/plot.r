args <- commandArgs(TRUE)

options(warn=-1)
#! /C/Program Files/R/R-3.4.1/bin/Rscript
suppressWarnings(suppressMessages(suppressPackageStartupMessages({
    library(dplyr,warn.conflicts = TRUE, quietly = TRUE)
    library(data.table,warn.conflicts = TRUE, quietly = TRUE)
    library(stringr,warn.conflicts = TRUE, quietly = TRUE)
    library(ggplot2,warn.conflicts = TRUE, quietly = TRUE)
    # library(plotly)
    # devtools::install_github('christophergandrud/networkD3')
    # library(igraph)
    library(RMySQL,warn.conflicts = TRUE, quietly = TRUE)
    library(pool,warn.conflicts = TRUE, quietly = TRUE)
})))

my_db <- dbPool(
  RMySQL::MySQL(), 
  user='reader', password='reader123', 
  host='139.224.15.11', 
#   host='127.0.0.1',
  dbname = 'aliyun_hive_data'
)

print(args[3])

get_read <- function(teacherid){  ## get_read('6740310') 
  pupils <- my_db %>% 
    tbl('qukan_member_info') %>% 
    filter(teacher_id == teacherid) %>% 
    collect()

  if(length(pupils$member_id)>0){
     a <- my_db %>% 
      tbl('qukan_bigid_read_v') %>% 
      filter(member_id %in% pupils$member_id) %>% 
      select(log_timestamp, member_id) %>%
      collect()
    
     b <- a %>%
           mutate(member_id = as.factor(member_id),
                  time = as.POSIXct(log_timestamp/1000, origin='1970-01-01 00:00:00')) %>% 
           tidyr::separate(time, c('thedate', 'thetime'), ' ') %>%
           filter(thedate == args[3])
     
     if(length(unique(b$member_id))>5 & length(unique(b$member_id)) < 50){
            img0 <-b %>%
                  select(-log_timestamp) %>% 
                  group_by(member_id) %>% 
                  arrange(thetime) %>% 
                  ggplot(aes(x = thetime, y = reorder(member_id, thetime, min))) +
                  geom_point()+
                  theme_void()+
                  # theme(axis.text = element_text(angle=90))
                  theme(axis.title = element_blank(),
                        axis.text = element_blank(),
                        legend.position = 'None')
        
      
      ggsave(path=args[2], filename=paste0(teacherid, '.jpg'), width=4, height=4)
      print('one image inserted.')
      
     }else{
       print('No read.')
       # print('No 3 pupils with ID > 100000000 read anything during the last three days.')
     }
    
  }else{
    # print('This member do not have a pupil.')
    print('No pupils.')
  }
}

# member <- fread('114.csv')
member <- fread(args[1],encoding = 'UTF-8')

sapply(member$teacher_id, get_read) %>% 
  data_frame(rlt = as.character(.)) %>% 
  group_by(rlt) %>% 
  tally()

poolClose(my_db)
