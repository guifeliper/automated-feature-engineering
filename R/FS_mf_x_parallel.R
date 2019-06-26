library(parallel)
library(mfe)

setwd('/Users/guifeliper/Thesis/Docker Python/python/data')
# Definition of datasets
datasets <- c(
  #  'abalone',
  # 'aids',
  # 'airlines',
  # 'allbp',
  # 'allrep',
  # 'analcatdata_creditscore',
  # 'authorship',
  # 'autoMpg',
  # 'autos',
  # 'autoUniv-au7-700',
  # 'backache',
  # 'balance-scale',
  'banana'
  # 'banknote-authentication',
  # 'blood-transfusion-service-center',
  # 'BNG_breast-w',
  # 'BNG_cmc',
  # 'bondrate',
  # 'breast-tissue',
  # 'breast-w',
  # 'calendardow',
  # 'cardiotocography',
  # 'churn',
  # 'climate-model-simulation-crashes',
  # 'cmc',
  # 'credit-approval',
  # 'credit-g',
  # 'cyyoung',
  # 'diabetes',
  # 'diabetes130US',
  # 'ecoli',
  # 'eeg-eye-state',
  # 'electricity',
  # 'engine',
  # 'eucalyptus',
  # 'first-order-theorem-proving',
  # 'glass',
  # 'haberman',
  # 'heart-statlog',
  # 'hepatitis',
  # 'hill-valley',
  # 'houses',
  # 'ilpd',
  # 'iris',
  # 'JapaneseVowels',
  #  'jm1',
  #  'jungle_chess_2pcs_raw_endgame_complete',
  #  'kc1',
  #  'kc2',
  #  'KDDCup09_upselling',
  #  'led-display-domain',
  # 'letter',
  # 'mammography',
  # 'morphological',
  # 'mozilla4',
  # 'oil_spill',
  # 'optdigits',
  # 'ozone-level-8hr',
  # 'page-blocks',
  # 'parkinsons',
  # 'pc1_req',
  # 'pc1',
  # 'pc3',
  # 'pc4',
  # 'pendigits',
  # 'phoneme',
  # 'prnn_crabs',
  # 'profb',
  # 'qsar-biodeg',
  # 'rmftsa_sleepdata',
  # 'satellite_image',
  # 'satellite',
  # 'satimage',
  # 'segment',
  # 'seismic-bumps',
  # 'smartphone-based_recognition_of_human_activities',
  # 'sonar',
  # 'spambase',
  # 'steel-plates-fault',
  # 'tae',
  # 'teachingAssistant',
  # 'thoracic-surgery',
  # 'thyroid-allbp',
  # 'thyroid-allhyper',
  # 'thyroid-allhypo',
  # 'thyroid-allrep',
  # 'thyroid-ann',
  # 'thyroid-dis',
  # 'titanic',
  # 'user-knowledge',
  # 'vehicle',
  # 'vinnie',
  # 'volcanoes-d4',
  # 'volcanoes',
  # 'wall-robot-navigation',
  # 'wdbc',
  # 'wholesale-customers',
  # 'wilt',
  # 'wine',
  # 'yeast'
)

mclapply(datasets, function(ds){
  metadata <- data.frame(read.csv(paste0("./filter_selection/", ds, ".csv"), header=TRUE, sep=","))
  rownames(metadata) <- make.names(metadata$feature, unique = TRUE)
  metadata <- metadata[ , -which(names(metadata) %in% c("X"))]
  
  
  print(paste("Processing ", ds, " dataset..."))
  X <- read.csv(paste0("../data/base/", ds, ".csv"), header=TRUE, sep=",")
  X_derived <- read.csv(paste0("../data/filter_extended/", ds, ".csv"), header=TRUE, sep=",")
  
  # Removing the base features
  dropList <- colnames(X)
  d.features <- X_derived[, !colnames(X_derived) %in% dropList]
  
  #Selecting class
  d.labels <- X[,'class']
  j = 1 
  numberOfFeatures = ncol(d.features)
  columns <- make.names(colnames(d.features))
  for (column in c(columns)){
    print(paste(j, 'of', numberOfFeatures, 'on', ds))
    d.info <- metafeatures(d.features[column], d.labels, groups=c("general"))
    position <- 1
    for (info in d.info)
    {
      name = paste0(tolower(names(d.info)[position]))
      metadata[c(column), c(name)] <- info
      position <- position + 1
    }
    j = j + 1
  }
  # Saving
  write.csv(metadata, file = paste0("../data/mfe_x/", ds,".csv"))
  
  
  ds
}, mc.cores=8L)

