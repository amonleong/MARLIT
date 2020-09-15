#Please intall carefully of of these packages in your R system
# 2020/09

install.packages("caret")
install.packages("lattice")
install.packages("DiagrammeR")
install.packages("pbapply")
install.packages("ggplot2")


if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("EBImage")

#mxnet
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")


# Alternatevelly:
#install.packages("https://github.com/jeremiedb/mxnet_winbin/raw/master/mxnet.zip", repos = NULL)

# AIImagePred must be installed from file
#shiny is installed only when running an app (change directory):
install.packages("AIImagePred_1.0.4.tar.gz", repos = NULL, type = "source")
