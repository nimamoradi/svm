# svm
svm classifier that will generate label by folder name 


this classifer will train and save clf trained parameters

## how to use 
this mdoel use folder name as label and it will find text files in it as train dataset 
 simply call model by this command 
 <pre>>python main.py "folder address"</pre>
 it will list folders and find txt files in it and will train model
 <br>
 by default it will use 10 percent of dataset as test and in the end will show it accuracy
 
 <br>
 to test model when model is trained you can simpl call 
 
<pre>>python model_use.py "folder address"</pre>
which will find label for each file in the folder address
