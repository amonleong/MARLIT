#llibreries
library(shiny)
library(AIImagePred) #Versió amb paràmetre my.opinion
library(grid)
library(ggplot2)
library(jpeg)
library(png)
library(mxnet)
library(pbapply)
library(exifr)
library(leaflet)

#Constants i dades fixes
train.images.dir <- paste(getwd(), "train",sep="/")
# Límit de pujada 10Mb en comptes del default de 5Mb
options(shiny.maxRequestSize = 10*1024^2)
#gràfics
himatge <- 300 #Altura de la imatge a la pantalla (només tab info)
colors <- c("yellow", "blue")
names(colors) <- c("Plastic", "No plastic")
#models
load("model400.RData")
model_default <- mx.unserialize(modelr)

#-------------------------------------------------------------

#funcions
extract_feature_test <- function(dir_path, 
                                 width, height, 
                                 is_plastic = TRUE, 
                                 add_label = TRUE) {
    img_size <- width*height
    ## List images in path
    images_names <- list.files(dir_path)
    #if (add_label) {
    ## Select only cats or dogs images
    #  images_names <- images_names[grepl(
            #ifelse(is_plastic, "plastic", "sin"), images_names)]
    ## Set label, plastic = 0, sin = 1
    label <- 0   #ifelse(is_plastic, 0, 1)
    #}
    print(paste("Start processing", length(images_names), "images"))
    ## This function will resize an image, turn it into greyscale
    feature_list <- pblapply(images_names, function(imgname) {
        ## Read image
        img <- readImage(file.path(dir_path, imgname))
        ## Resize image
        img_resized <- resize(img, w = width, h = height)
        ## Set to grayscale
        grayimg <- channel(img_resized, "gray")
        ## Get the image as a matrix
        img_matrix <- grayimg@.Data
        ## Coerce to a vector
        img_vector <- as.vector(t(img_matrix))
        return(img_vector)
    })
    ## bind the list of vector into matrix
    feature_matrix <- do.call(rbind, feature_list)
    feature_matrix <- as.data.frame(feature_matrix)
    ## Set names
    names(feature_matrix) <- paste0("pixel", c(1:img_size))
    #if (add_label) {
    ## Add label
    feature_matrix <- cbind(label = label, feature_matrix)
    #}
    return(feature_matrix)
}

predict.count <- function(model,
                          predict.test.images.dir,
                          name.files.images=c()){
    print("Read the test images........")
    mainDir <- predict.test.images.dir
    subDir <- "outputDirectory"
    image_dir<-file.path(mainDir, subDir)
    size_foto<-25
    plastic_data <- extract_feature_test(dir_path = image_dir, 
                                         width = size_foto, 
                                         height = size_foto,
                                         add_label = T) #SIEMPRE ETIQUETA A 1
    ## Bind rows in a single dataset
    complete_set <- rbind(plastic_data)
    test_set <- complete_set
    dim(test_set)
    
    test_data <- data.matrix(test_set)
    test_x <- t(test_set[,-1])
    test_y <- test_set[,1]
    test_array <- test_x
    dim(test_array) <- c(size_foto, size_foto, 1, ncol(test_x))
    ##########################################################################
    ## AQUI HACE LA PREDICCION DE LAS IMAGENES (0 = PLASTIC, 1 = SIN PLASTIC)
    predict_probs <- predict(model, test_array)
    predicted_labels <- max.col(t(predict_probs)) - 1
    images_names <- list.files(image_dir)
    
    results.clase.images.predicted<-data.frame(images_names, predicted_labels)
    a<-data.frame(results.clase.images.predicted[1])
    #contar numero de imagenes de cada  tipo
    resultat.complet.imatge <- 
        100*table(a$predicted_labels)/length(a$predicted_labels)
    #estructura del retorn per backcompatibilitat (s'ha de simplificar)
    return(list(resultat.complet.imatge, 
                list(results.clase.images.predicted,
                     NA,
                     "class: plastic = 0, sin = 1")))
}

#funcion para ajustar modelo para predecir plasticos 
#a partir de unas imagenes divididas en 0-plastic, 1-non.plastic
model.class.DL.plastic<- function(train.images.dir, 
                                  num.round = 150,
                                  test=FALSE){
    
    #parametros del modelo, es de la matriz de trabajo de 25 x 25 pixeles
    size_foto<-25
    
    setwd(train.images.dir)
    #imagenes del training donde se han extraido sus caracteristicas y son matrices B/N 25*25
    plastic_data<-readRDS("plastic.rds", refhook = NULL)
    sin_data<-readRDS("sin.rds", refhook = NULL)
    
    ## Bind rows in a single dataset
    set.seed(1078) #fix random seed
    complete_set <- rbind(plastic_data, sin_data)
    ## test/training partitions (80% training, 20% test)
    training_index <- createDataPartition(complete_set$label, p = .9, times = 1)
    training_index <- unlist(training_index)
    train_set <- complete_set[training_index,]
    dim(train_set)
    ## [1] 22500   785
    test_set <- complete_set[-training_index,]
    dim(test_set)
    ## [1] 2500  785
    #Reshape the data into a proper format required by the model:
    
    ## Fix train and test datasets
    train_data <- data.matrix(train_set)
    train_x <- t(train_data[, -1])
    train_y <- train_data[,1]
    train_array <- train_x
    dim(train_array) <- c(size_foto, size_foto, 1, ncol(train_x))
    
    test_data <- data.matrix(test_set)
    test_x <- t(test_set[,-1])
    test_y <- test_set[,1]
    test_array <- test_x
    dim(test_array) <- c(size_foto, size_foto, 1, ncol(test_x))
    #Training the model:
    
    library(mxnet)
    ## Model
    mx_data <- mx.symbol.Variable('data')
    ## 1st convolutional layer 5x5 kernel and 20 filters.
    conv_1 <- mx.symbol.Convolution(data = mx_data, kernel = c(5, 5), num_filter = 50)
    tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
    pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2,2 ))
    ## 2nd convolutional layer 5x5 kernel and 50 filters.
    conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5,5), num_filter = 50)
    tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
    pool_2 <- mx.symbol.Pooling(data = tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
    ## 1st fully connected layer
    flat <- mx.symbol.Flatten(data = pool_2)
    fcl_1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 100)
    tanh_3 <- mx.symbol.Activation(data = fcl_1, act_type = "tanh")
    ## 2nd fully connected layer
    fcl_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 10)
    ## Output
    NN_model <- mx.symbol.SoftmaxOutput(data = fcl_2)
    
    ## Set seed for reproducibility
    mx.set.seed(196)
    
    ## Device used. Sadly not the GPU :-(
    device <- mx.cpu()
    
    print("Calculating the DL model. Please patient!!!!!!!........")
    ## Train on 1200 samples
    #num.round=250
    model <- mx.model.FeedForward.create(NN_model, X = train_array, y = train_y,
                                         ctx = mx.cpu(),
                                         num.round = num.round,
                                         momentum = 0.9,
                                         array.batch.size = 100,
                                         learning.rate = 0.025,
                                         wd = 0.00001,
                                         eval.metric =mx.metric.accuracy,
                                         epoch.end.callback = 
                                             mx.callback.log.train.metric(100))
    accuracy.modelDL<-NA
    if(test) {
        ## train set
        predict_probs <- predict(model, train_array)
        predicted_labels <- max.col(t(predict_probs)) - 1
        table(train_data[, 1], predicted_labels)
        #accuracy
        accuracy.modelDL<- 
            sum(diag(table(train_data[, 1], predicted_labels)))/
            sum(table(train_data[, 1], predicted_labels))
        print(paste("Accuracy DL model train set= ", 
                    round(accuracy.modelDL,3), sep =""))
        
        
        
        ## Test test set
        predict_probs <- predict(model, test_array)
        predicted_labels <- max.col(t(predict_probs)) - 1
        table(test_data[, 1], predicted_labels)
        #accuracy
        accuracy.modelDL<- 
            sum(diag(table(test_data[, 1], predicted_labels)))/
            sum(table(test_data[, 1], predicted_labels))
        print(paste("Accuracy DL model test set (at random 10%)= ", 
                    round(accuracy.modelDL,3), sep =""))
    } 
    return(list(model, accuracy.modelDL))
}

#Extreu dades d'exif
netejaexif <- function(dades) {
    noms <- names(dades)
    if ("Lens35efl" %in% noms) {
        f35 <- dades$Lens35efl
    } else {
        f35 <- dades$FocalLength35efl
    }
    return(list(alt=dades$GPSAltitude,
                lat=dades$GPSLatitude,
                lon=dades$GPSLongitude,
                f35=f35,
                hora=dades$DateTimeOriginal))
}

#----------------------------------------------------------------
# Define UI for application 
ui <- fluidPage(
    
    titlePanel("Identification of floating litter from aerial imaging"),
    
    sidebarLayout(
        sidebarPanel(
            fileInput("fitxer",
                      "Upload jpg file to analyze:",
                      multiple=FALSE),
            actionButton("analitza", "Analyze image"),
            numericInput("numsplit", 
                         "Image splitting (rows and columns)",
                         value=1),
            numericInput("altura","Viewpoint height (m)",
                         value=200),
            numericInput("dfocal",
                         "Focal distance equivalent to 24x36 mm (mm)",
                         value=40),
            checkboxInput("clickexif", 
                          "Read information from image metadata",
                          value=TRUE),
            textOutput("model_status"),
            actionButton("model_default", "Use default model"),
            actionButton("model_train", "Train new model"),
            numericInput("numround", 
                         "Iterations of new model",
                         value=200),
            checkboxInput("test_accuracy", 
                          "Test new model accuracy"),
            downloadButton("model_save", "Download current model"),
            fileInput("model_load",
                      "Upload model file:",
                      multiple=FALSE),
            img(src='LOGO_COLOUR_MEDSEALITTER.jpg', 
                width="220"),
            img(src='marca_pos_rgb.jpg', 
                width="220"),
            br(),
            p(br(),strong("Developed by:"), br(),
              "Pere López Brosa", br(), "Antonio Monleón-Getino", br(),
              "BIOST3 - ",
              tags$a("http://www.fbg.ub.edu/en/researchers/research-groups-of-the-ub/estadistica-clinica-biodiversitat/",
              href="http://www.fbg.ub.edu/en/researchers/research-groups-of-the-ub/estadistica-clinica-biodiversitat/",
              style="font-size:90%"), style="font-size:90%"
            )
            
            
            #downloadButton("baixa", "Download anàlisi")
        ),
        
        mainPanel(
            tabsetPanel(
                tabPanel("Results",
                         strong(textOutput("status")),
                         textOutput("nomfitxer"),
                         textOutput("area"),
                         textOutput("densitat"),
                         textOutput("altitud"),
                         plotOutput("grafcomplet"),
                         #plotOutput("grafsimple"),
                         plotOutput("grafbarres")
                ),
                tabPanel("Image info",
                         plotOutput("imatge", height = himatge),
                         br(),
                         p("Information according to image Exif data:"),
                         verbatimTextOutput("exifinfo"),
                         leafletOutput("mapa")
                         ),
                tabPanel("Help",
                         h1("Getting started"),
                         p("Load a .jpg image to be identified."),
                         p("If you want the image to be divided in pieces
                           before analyzing, select in field 'Image splitting'
                           the number of rows and columns."),
                         p("Click 'Analyze image' to start analysis."),
                         h1("Image area"),
                         p("The sea area included in the image
                           is computed using lens focal lenght
                           and viewpoint height. Focal lenght
                           is assumed to be equivalent to a 24x36 mm 
                           sensor."),
                         h1("Managing models"),
                         p("The app uses a model trained in advance
                           using deep learning and a built in training set.
                           It is possible to train other models with the 
                           same set, and to save and retrieve them.")
                ),
                tabPanel("Raw results",
                         p("Raw results of package AIIpred:"),
                         verbatimTextOutput("sortida3"),
                         p("Other results intended for debugging:"),
                         textOutput("sortida"),
                         verbatimTextOutput("sortida2"),
                         verbatimTextOutput("sortida4"),
                         verbatimTextOutput("sortida5"),
                         verbatimTextOutput("sortida6"),
                         verbatimTextOutput("sortida7")
                )
            )
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
    
    res <- reactiveValues(
        respred=NA,
        resvalid=FALSE
    )
    
    model <- reactiveValues(
        model=model_default,
        status="Using default model"
    )
    
    # PER FER: en comptes de [1] hauria de ser unique()
    # per cobrir el cas de múltiples directoris temporals
    directori <- reactive({
        unlist(
            lapply(
                strsplit(input$fitxer$datapath, "/"),
                function(x) paste(x[1], x[2], sep="/"))[1])
    })
    
#    observe({
#        if (length(input$fitxer$datapath)>0){
#            nomvell <- input$fitxer$datapath[
#                grep("JPG$",input$fitxer$datapath)]
#            nomnou <- gsub("JPG$","jpg",nomvell)
#            file.rename(nomvell, nomnou)}
#    })

    observeEvent(input$model_default,{
        model$model <- model_default
        model$status <- "Using default model"
    })

    observeEvent(input$model_load$name, {
        if (grepl("rdata$", tolower(input$model_load$name))) {
            nommodel <- load(input$model_load$datapath)
            model$model <- mx.serialize(get(nommodel))
            model$status <- paste("Using model", 
                                  nommodel, 
                                  "from file", 
                                  input$model_load$name)
        } else {
            if (grepl("rds$", tolower(input$model_load$name))) {
                model$model <- 
                    mx.serialize(readRDS(input$model_load$datapath))
                model$status <- paste("Using model from file", 
                                      input$model_load$name)
            } else {
                showNotification("Only .RData and .rds files.", 
                                 duration = 18,
                                 type="error")
            }
        }
    }) 

    observeEvent(input$model_train, {
        showNotification("Fitting model. Please wait a few minutes.", 
                         duration = NULL,
                         id="ajustant_model")
        res <- model.class.DL.plastic(train.images.dir, 
            num.round = input$numround,
            test=input$test_accuracy)
        model$model <- res[[1]]
        model$status <- paste("Using trained model.",
                              ifelse(is.na(res[[2]]),
                                     "",
                                     paste("Test accuracy:", 
                                           round(res[[2]],3))))
        removeNotification(id="ajustant_model")
    }) 

    output$model_save <- downloadHandler(
        filename="allplast_model.rds",
        content=function(file) {
            saveRDS(mx.serialize(model$model), file = file)
        }
    )
                
    observe({
        if (!is.null(input$fitxer$name)) {
            res$resvalid <- FALSE
            res$respred <- NA
            print("Esborrant fitxers auxiliars")
            dout <- paste0(directori(), "/outputDirectory")
            fitxers <- dir(dout)
            fitxers <- file.path(dout, fitxers)
            file.remove(fitxers)
            print("Esborrats fitxers auxiliars")
        }
    })
    
    observeEvent(input$numsplit,{
            if (res$resvalid) {
                res$resvalid <- FALSE
                res$respred <- NA
                print("Esborrant fitxers auxiliars")
                dout <- paste0(directori(), "/outputDirectory")
                fitxers <- dir(dout)
                fitxers <- file.path(dout, fitxers)
                file.remove(fitxers)
                print("Esborrats fitxers auxiliars")
            }
    })
    
    observeEvent(input$analitza, {
        if (is.null(input$fitxer$name)) {
            showNotification("An image needs to be uploaded 
                             before analyzing",
                             type="error")
        } else {
            print("Calculant")
            showNotification(paste("Splitting image in",(input$numsplit)^2,"pieces."),
                             duration=NULL, id="dividint")
            res$resvalid <- FALSE
            name.files.images<-c("")
            print(paste("Separant en", input$numsplit,"files i columnes"))
            print(paste("Buscant al directori:", directori()))
            if (!dir.exists(file.path(directori(), 
                                     "outputDirectory"))) {
                dir.create(file.path(directori(), 
                                     "outputDirectory"))
            }
            image.splitting.algorithm(dir.imag=directori(), 
                                      dir.imag.final=file.path(directori(), 
                                                               "outputDirectory"), 
                                      n.div=input$numsplit)
            removeNotification(id="dividint")
            showNotification("Analyzing image pieces.",
                             duration=NULL, id="calculant")
            res$respred <- predict.count(model = model$model,
                                         predict.test.images.dir=directori()
                                         )
            res$resvalid <- TRUE
            print("Càlcul acabat")
            removeNotification(id="calculant")
        }
        
    }) 

    resultat <- 
        reactive({
            if(is.list(res$respred)) {
                showNotification("Analysis done. Preparing results.", 
                                 duration = NULL,
                                 id="analitzant_resultats")
                print("Analitzant el resultat")
                res0 <- res$respred[[2]][[1]]
                names(res0) <- c("nom","plastic")
                res0$nom <- as.character(res0$nom)
                res0$num <- 
                    as.numeric(
                        sapply(strsplit(res0[[1]], "_", fixed=TRUE),`[`,2))
                n <- sqrt(length(res0$num))
                res0$col <- (res0$num-1)%/%n+1
                res0$fila <- (res0$num-1) %% n+1
                res0$plastic <- 
                    factor(res0$plastic, 
                           levels=c(0,1),
                           labels=c("Plastic", "No plastic"))
                print("Anàlisi acabat")
            } else {
                res0 <- data.frame(nom=NA, plastic=NA, fila=NA, col=NA)
            }
            removeNotification(id="analitzant_resultats")
            res0
        })
    
    imatge <- reactive({
        if (grepl("png$", input$fitxer$name)) {
            imatge0 <- readPNG(input$fitxer$datapath)
        } else {
            imatge0 <- readJPEG(input$fitxer$datapath)
        }
        imatge0
    })
    
    rawexif <- reactive(
        if(!is.null(input$fitxer$datapath)) {
            read_exif(input$fitxer$datapath)
        }
    )

    exifnet <- reactive({
        netejaexif(rawexif())
    })
    

    relaspecte <- reactive({
        if (!is.null(input$fitxer$name)) {
            im<-imatge()
            r <- dim(im)[2]/dim(im)[1]
        } else {
            r <- 4/3
        }
        r
    })
        
    area <- reactive({
        if (input$clickexif) {
            (exifnet()$alt/exifnet()$f35)^2*(36^2/relaspecte())
        } else {
            (input$altura/input$dfocal)^2*(36^2/relaspecte())
        }
    })
    
    output$model_status <- renderText(
        model$status
    )
        
    output$nomfitxer <-renderText({
        if (length(input$fitxer$name)==0) {
            "Please upload an image to analyze."
        } else {
            paste("Image:",input$fitxer$name)
        }
    })    
    
    output$area <- renderText({
        if (length(input$fitxer$name)>0) {
            paste("Area covered by image:",
                  round(area(),2),
                  "m2 (",
                  round(area()/1e4,4),
                  "Ha)")
        } else {
            ""
        }
    })

    output$altitud <- renderText({
        if (length(input$fitxer$name)>0 & !is.null(exifnet()$alt)) {
            paste("Altitude of viewpoint:",
                  round(exifnet()$alt,1),
                  "m")
        } else {
            ""
        }
    })
    
    output$densitat <-renderText({
        if (length(input$fitxer$name)==0 | !res$resvalid) {
            ""
        } else {
            nplast <- sum(resultat()$plastic=="Plastic")
            tplast <- length(resultat()$plastic)
            sortida <- paste("Detected",
                             nplast,
                             "cells with litter out of",
                             input$numsplit^2, "cells.\n",
                             round(nplast/area()*1e6,2),
                             "pieces/Km2")
            sortida
        }
    })    

    output$status <- renderText({
        if (length(input$fitxer$name)==0) {
            "Please load an image."
        } else {
            if (res$resvalid) {
                ""
            } else {
                "Press 'Analyze image' to start analysis."
            }
        }
            })
    
    output$grafcomplet <- renderPlot({
        if (!res$resvalid) {
            if (length(input$fitxer$name)==0) {
                plot.new()
            } else {
                ggplot(data=resultat()) +
                    annotation_custom(rasterGrob(imatge(),
                                                 width=unit(1,"npc"),
                                                 height=unit(1, "npc")),
                                      -Inf, Inf, -Inf, Inf)
            }          
        } else {
            ggplot(data=resultat()) +
                annotation_custom(rasterGrob(imatge(),
                                             width=unit(1,"npc"),
                                             height=unit(1, "npc")),
                                  -Inf, Inf, -Inf, Inf)+
                geom_point(aes(x=col, y=fila, color=plastic),
                           size=5) +
                scale_y_reverse(name=NULL,
                                limits=c(input$numsplit+.5, .5)) +
                xlim(c(.5, input$numsplit+.5)) +
                xlab(NULL) +
                scale_color_manual(values=colors)
        }
    })
    

    output$grafsimple <- renderPlot({
        if (!res$resvalid) {
            plot.new()
        } else {
            ggplot(data=resultat()) +
                geom_point(aes(x=col, y=fila, color=plastic),
                           size=5) +
                scale_y_reverse() +
                scale_color_manual(values=colors)
        }
    })
    
    output$grafbarres <- renderPlot({
        if (!res$resvalid) {
            plot.new()
        } else {
            ggplot(data=resultat()) +
                geom_bar(aes(x=plastic, fill=plastic)) +
                scale_fill_manual(values=colors)
        }
    })

    #Outputs tab image info
        
    output$imatge <- renderImage({
        outfile <- tempfile(fileext = '.jpg')
        if (!is.null(input$fitxer$name)) {
            im<-imatge()
        } else {
            im<-array(c(1,.5),dim=c(4,3,3))
        }
        writeJPEG(im, outfile)
        list(src = outfile,
             contentType = 'image/jpg',
             width = dim(im)[2]/dim(im)[1]*himatge,
             height = himatge,
             alt = "Imatge original")
    }, deleteFile = TRUE)
    
    output$exifinfo <- renderPrint({
        cat(paste("File name:",
                  input$fitxer$name,"\n"))
        if (!is.null(input$fitxer$name)) {
            cat(paste("Image size:",
                      dim(imatge())[2],"x",
                      dim(imatge())[1],"pixels\n"))
        }
        cat(paste("Focal distance (35 mm equivalent):", 
                  exifnet()$f35,"mm\n"))
        cat(paste("Altitude:", 
                  exifnet()$alt,"m\n"))
        cat(paste("Latitude:", 
                  exifnet()$lat,"\n"))
        cat(paste("Longitude:", 
                  exifnet()$lon,"\n"))
        cat(paste("Area covered by image:", 
                  round(exifnet()$alt/exifnet()$f35*36,2),"x",
                  round(exifnet()$alt/exifnet()$f35*36/relaspecte(),2),
                  "m\n"))
        cat(paste("Area covered by image:", 
                  round(exifnet()$alt/exifnet()$f35*36*
                            exifnet()$alt/exifnet()$f35*36/relaspecte(),
                        2),"m2\n"))
        cat(paste("Taken on", exifnet()$hora,"\n"))
    })
    
    output$mapa <- renderLeaflet({
        mapa <- leaflet()
        mapa <- addTiles(mapa)
        if(!is.null(exifnet()$lon)) {
            mapa <- addMarkers(mapa, 
                               lng=exifnet()$lon, 
                               lat=exifnet()$lat)
        }
        mapa
    })
    
    # Outputs tab auxiliar
    
    output$sortida <- renderText({
        print(input$fitxer$datapath)
    })
        
    output$sortida2 <- renderPrint({
        print(input$fitxer)
        print(directori())
        print(dir(directori()))
        print(getwd())
        print(train.images.dir)
    })
    output$sortida3 <- renderPrint({
        print("res$respred:")
        print(res$respred)
    })
    
    output$sortida4 <- renderPrint({
        print("resultat():")
        print(resultat())
    })

    output$sortida5 <- renderPrint({
        print(is.null(imatge()))
        #print(str(imatge))
    })
    
    output$sortida6 <- renderPrint({
        print(train.images.dir)
        print(dir(train.images.dir))
        print("directori()")
        print(directori())
        print("dir(directori())")
        print(dir(directori()))
        print("getwd():")
        print(getwd())
        print("system.file():")
        print(system.file())
    })
    
    output$sortida7 <- renderPrint({
        print("exifnet()")
        print(exifnet())
        print("rawexif():")
        print(str(rawexif()))
    })
    
    
}

# Run the application 
shinyApp(ui = ui, server = server)
