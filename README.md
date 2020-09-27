# MARLITE

Shiny app for floating marine litter detection in aerial images.

This directory contains the instructions and software needed to install the Shiny MARLIT 
application on a local computer (Windows Operating System). 

Detailed information on the application functioning and use is illustrated in:
    Automatic detection and quantification of floating marine macro-litter in aerial images: introducing 
    a novel deep learning approach connected to a web application in R
    By: Odei Garcia-Garin, Toni Monleón-Getino, Pere López-Brosa, Asunción Borrell, Alex Aguilar, 
    Ricardo Borja-Robalino, Luis Cardona, Morgana Vighi

The study describes the development of a CNN-based algorithm for the detection of floating marine 
macro litter in aerial images and its connection to a web-oriented application based on the Shiny 
package.

To use the application, it is necessary to have R (version 3.6) and Rstudio installed.

Follow the installation instructions described in the instal_paquets.R file and execute them in R.

Run the shiny application contained in the AllPlast directory from Rstudio, as it is usually done with 
Shiny applications (see https://rstudio.com/resources/webinars/introduction-to-shiny/)

Inside the test IMAGES folder you will find different images with and without floating litter that 
can be used to carry out tests. Otherwise, you can test the application by using your own aerial 
images, provided they respond to the requirements described in the study (images should be taken 
in the nadir position and the ground sampling distance should be at least 3.6 cm).

Please, contact us at odei.garcia@ub.edu or amonleong@ub.edu whether you have any 
doubts/comments regarding the functioning of the application, or in case you have any suggestions 
for its improvement or you are interested in sharing any set of aerial images of the sea surface.

