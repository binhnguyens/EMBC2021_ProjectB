%% DEMO FILE
clear all
close all
clc;

%% Load the data and select features for classification
inputs =[1,4:8,10:19];

fileID = fopen('FT_analysis.txt','w');

for i = 1:length (inputs)
    
    % Input of the different types of algorithms
    usr_input = inputs(i);
    
    % Run the program
    Demo_modified_jan22;
    pause(1);
    
    fprintf(fileID,'Selection method: %s\n', selection_method);
    fprintf(fileID,'DT Acc: %.2f\n', ((1-err_DT)*100));
    fprintf(fileID,'SVM Acc: %.2f\n\n', ((1-err_SVM)*100));
    
end

fclose(fileID);