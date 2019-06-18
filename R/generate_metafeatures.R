#!/usr/bin/Rscript

# install.packages(c("mfe"), dependencies=TRUE, repos='http:/cran.rstudio.com/)
library(mfe)
library(parallel)


setwd('/Users/guifeliper/Thesis/autoML/automated-feature-engineering/R/Data')

# Definition of datasets
datasets <- list.files('./base/',full.names = FALSE)

# Get meta knowledge
metadata <- read.csv(paste(sep="", "./metadata.csv"), header=TRUE, sep=",", row.names = 1)


#Process the metafeatures
creating_metafeatures <- function(ds, type, groups) {
  print(paste("Processing", type,  "for", ds, "..."))
  d <- read.csv(paste0("./", type , "/", ds), header=TRUE, sep=",")
  
  d.features <- subset(d, select=-c(class))
  d.labels <- d[,'class']
  
  d.info <- tryCatch(
    metafeatures(d.features, d.labels, groups=groups),
    error=function(e) {print(paste(e));NaN}
  )
  
  i = 1 
  for (info in d.info)
  {
    name = paste0(type,".", tolower(names(d.info)[i]))
    metadata[c(ds), c(name)] <- info
    i = i + 1
  }
  
  return(metadata)
}





# Process datasets
# Parallel only for Linux and OS X
mclapply(datasets, function(ds){
  start.time <- Sys.time()
  metadata<- creating_metafeatures(ds, 'extended', c("general", "statistical","model.based"))
  metadata<- creating_metafeatures(ds, 'base', c("general", "statistical","model.based"))
  gc()

  #Printing
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  print(time.taken)
  ## Saving
  write.csv(metadata, file = "./metadata.csv")
}, mc.cores=8L)

