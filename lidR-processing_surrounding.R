#Load necessary libraries
library("lidR")
library("lmom")
library("sp")
library("terra")
library("dplyr")
library("tools")
library("raster")
#Set working directory
setwd("C:/Users/Dag Bjornberg/AI/ForestMap/Latvian_Image_data")
#read a las file
las <- readLAS("las_surrounding/2444512333.las")
#normalize las file
las_normalized <- normalize_height(las, knnidw())
xy_coords <- las_normalized@data[, c("X", "Y")]

# Calculate the mean of X and Y coordinates
min_x <- min(xy_coords$X)
max_x <- max(xy_coords$X)
min_y <- min(xy_coords$Y)
max_y <- max(xy_coords$Y)
mid_x <- (max_x + min_x)/2
mid_y <- (max_y + min_y)/2

mean_x <- mean(xy_coords$X)
mean_y <- mean(xy_coords$Y)

# Normalize the X and Y coordinates by subtracting the means
las_normalized@data$X <- las_normalized@data$X - mid_x
las_normalized@data$Y <- las_normalized@data$Y - mid_y

plot(las_normalized, axis = TRUE, bg = "white")


folder_path <- "las_surrounding"
# Get the list of files in the folder
file_names <- list.files(folder_path)
las_files <- file_names[grepl("\\.las$", file_names)]
#output folder for processed las files
output_folder <- "las_surrounding_processed"
output_folder_tif <- "tif_surrounding_processed"
for (file_name in las_files){
  # Full path to the file
  full_path <- file.path(folder_path, file_name)
  plot_identifier = file_path_sans_ext(file_name)
  tryCatch({
  las <- readLAS(full_path) 

  las_normalized <- normalize_height(las, knnidw())
  las_norm_tosave <- las_normalized
  # Extract X and Y coordinates
  xy_coords <- las_normalized@data[, c("X", "Y")]

  # Calculate the mean of X and Y coordinates
  min_x <- min(xy_coords$X)
  max_x <- max(xy_coords$X)
  min_y <- min(xy_coords$Y)
  max_y <- max(xy_coords$Y)
  mid_x <- (max_x + min_x)/2
  mid_y <- (max_y + min_y)/2

  mean_x <- mean(xy_coords$X)
  mean_y <- mean(xy_coords$Y)

  # Normalize the X and Y coordinates by subtracting the means
  las_normalized@data$X <- las_normalized@data$X - max_x
  las_normalized@data$Y <- las_normalized@data$Y - max_y

  las_norm_tosave@data$X <- las_norm_tosave@data$X - mid_x
  las_norm_tosave@data$Y <- las_norm_tosave@data$Y - mid_y

  # Filter heighs above 40m
  # Extract Z coordinates from the LAS data using @data
  z_coords <- las_normalized@data[["Z"]]  # Access Z values
  z_coords_tosave <- las_norm_tosave@data[["Z"]]
  # Specify the Z value threshold (e.g., 40 meters)
  z_threshold <- 40

  # Filter out points with Z > threshold
  las_normalized <- las_normalized[z_coords <= z_threshold, ]
  las_norm_tosave <- las_norm_tosave[z_coords_tosave <= z_threshold, ]
  #compute metrics for normalized las file
  first_metrics <- pixel_metrics(las_normalized, func=.stdmetrics, 10)
  second_metrics <- pixel_metrics(las_normalized, func = ~as.list(lmom::samlmu(Z)), 10) 
  combined_metrics <- c(first_metrics, second_metrics)
  output_file <- file.path(output_folder, basename(file_name))
  writeLAS(las_norm_tosave , output_file)
  #compute metrics for normalized las file
  first_metrics <- pixel_metrics(las_normalized, func=.stdmetrics, res = 10)
  second_metrics <- pixel_metrics(las_normalized, func = ~as.list(lmom::samlmu(Z)), res=10) 
  combined_metrics <- c(first_metrics, second_metrics)
  tif_file <- file.path(output_folder_tif, paste0(plot_identifier, ".tif"))
  print(tif_file)
  terra::writeRaster(combined_metrics, tif_file, filetype = "GTiff", overwrite = TRUE)
    
  }, error = function(e) {
    warning(paste("Error processing file:", file_name, "-", e$message))
  })
  
}
 








