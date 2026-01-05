# Custom-Nail-Solutions
This repository contains all code that is written/used in the "Custom Nail Solutions" senior design project. Contains code from the librealsense public repository. 

All important code is located within the "NailFinderWorkingCrop" folder, where the "Custom_Nail_Solutions_App" is saved, as well as other smaller scripts that need to be integrated into a single pipeline.

The "Custom_Nail_Solutions_App" is the primary software that runs on the box that is in charge of displaying the current image in real-time, capturing the point clouds of hands, operating the servo, handling all the data manipulation to save the point clouds, and displaying the point clouds if desired. The current process requires a user to then take those saved point cloud files, and uploading them manually to whichever file-sharing system is desired (the senior design team used BOX to share files between each other). 

Additionally, some of the other scripts (Combined TSDF Script, Data Analysis, File Checker, Point Clouder Helper, RDBD_Odometry, Solve Displacement Vector, etc...) are either obsolete or are currently used but have not been integrated into the Custom_Nail_Solutions_App. 

The other code located in this repository is primarily code from the open-source library of librealsense from Intel, and is located in this repository such that our code can access it from our developer boards of Orange Pi / Raspberry Pi
