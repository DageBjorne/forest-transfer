#Load necessary libraries
library("lidR")
library("lmom")
library("sp")
library("terra")
library("dplyr")
library("tools")
#Set working directory
setwd("C:/Users/Dag Bjornberg/AI/ForestMap/Latvian_Image_data")
#read a las file
las <- readLAS("las/2444512333.las")
#normalize las file
las_normalized <- normalize_height(las, knnidw())
#compute metrics
first_metrics <- cloud_metrics(las_normalized, func = .stdmetrics)
second_metrics <- cloud_metrics(las_normalized, func = ~as.list(lmom::samlmu(Z))) 
combined_metrics = c(first_metrics, second_metrics)
combined_metrics_df <- as.data.frame(t(combined_metrics))
# Initialize dataframe with correct columns
las_normalized <- normalize_height(las, knnidw())
first_metrics <- cloud_metrics(las_normalized, func = .stdmetrics)
second_metrics <- cloud_metrics(las_normalized, func = ~as.list(lmom::samlmu(Z)))
combined_metrics <- c(first_metrics, second_metrics)
df <- data.frame(matrix(ncol = length(names(combined_metrics)), nrow = 0))
colnames(df) <- names(combined_metrics)
#create new column to store the filename
df <- df %>% mutate(Plot = NA) %>% relocate(Plot, .before = everything())
# Define the folder path
folder_path <- "las"
# Get the list of files in the folder
file_names <- list.files(folder_path)
las_files <- file_names[grepl("\\.las$", file_names)]
df <- data.frame()

#output folder for processed las files
output_folder <- "las_processed"
for (file_name in las_files){
  # Full path to the file
  full_path <- file.path(folder_path, file_name)
  plot_identifier = file_path_sans_ext(file_name)
  tryCatch({
  las <- readLAS(full_path) 
  las_normalized <- normalize_height(las, knnidw())
  # Extract X and Y coordinates
  xy_coords <- las_normalized@data[, c("X", "Y")]

  # Calculate the mean of X and Y coordinates
  mean_x <- mean(xy_coords$X)
  mean_y <- mean(xy_coords$Y)

  # Normalize the X and Y coordinates by subtracting the means
  las_normalized@data$X <- las_normalized@data$X - mean_x
  las_normalized@data$Y <- las_normalized@data$Y - mean_y

  # Filter heighs above 40m
  # Extract Z coordinates from the LAS data using @data
  z_coords <- las_normalized@data[["Z"]]  # Access Z values

  # Specify the Z value threshold (e.g., 40 meters)
  z_threshold <- 40

  # Filter out points with Z > threshold
  las_normalized <- las_normalized[z_coords <= z_threshold, ]
  # save the filtered las file
  output_file <- file.path(output_folder, basename(file_name))
  writeLAS(las_normalized , output_file)
  #compute metrics for normalized las file
  first_metrics <- cloud_metrics(las_normalized, func = .stdmetrics)
  second_metrics <- cloud_metrics(las_normalized, func = ~as.list(lmom::samlmu(Z)))
  combined_metrics = c(first_metrics, second_metrics)
  combined_row = c(plot_identifier, combined_metrics)
  combined_metrics_df <- as.data.frame(t(combined_row))
  
  df <- bind_rows(df, combined_metrics_df)
    
  }, error = function(e) {
    warning(paste("Error processing file:", file_name, "-", e$message))
  })
  
}



df <- data.frame(lapply(df, function(column) {
  if (is.list(column)) {
    # Collapse list elements into a single string
    sapply(column, function(x) as.numeric(x[1]))
  } else {
    column  # Leave other columns unchanged
  }
}), stringsAsFactors = FALSE)


#rename first column again (TODO: fix this bug!)
colnames(df)[1] <- "Plot"

write.csv(df, "lidr-metrics.csv", row.names = FALSE, quote = FALSE)
# Read the CSV file
df_check <- read.csv("lidr-metrics.csv")

# Print the first few rows of the dataframe
head(df_check)



