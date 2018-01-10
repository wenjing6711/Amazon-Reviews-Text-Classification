library(tm)
library(SnowballC)
library(wordcloud)
library(ggplot2)
## TVs, laptops, cameras, tablets, mobilephone, video_surveillance
#filename = "/Users/wenjing/Box Sync/UIUC/3rd/FALL 2017/CS 510/Project/single_category/titles.csv"
filename = "/Users/wenjing/Box Sync/UIUC/3rd/FALL 2017/CS 510/Project/ind_category/titles_video_surveillance.csv"
jeopQ <- read.csv(filename, stringsAsFactors = FALSE)
jeopCorpus <- Corpus(VectorSource(jeopQ))
jeopCorpus <- tm_map(jeopCorpus, content_transformer(tolower))
#jeopCorpus <- tm_map(jeopCorpus, removePunctuation)
#jeopCorpus <- tm_map(jeopCorpus, stemDocument)
pal2 <- brewer.pal(8,"Dark2")
setEPS()
postscript("word_cloud_titles_video_surveillence.eps")
wordcloud(jeopCorpus,scale=c(5,0.4), max.words = 200, random.order = FALSE, colors = pal2)
dev.off()
