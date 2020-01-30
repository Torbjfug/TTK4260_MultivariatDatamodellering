readFile = false;
if(readFile)
    file_path = "Data/PowerProduction/";
    X = csvimport(convertStringsToChars(file_path + "X_train.csv"),'delimiter',';');
    
    Y = csvimport(convertStringsToChars(file_path + "Y.csv"),'delimiter',';');
    Y = Y(:,2);
end

header = X(1,:)



X = X(:,2:);