try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter
import time
import numpy as np
from pprint import pprint
import os
import pathlib
import csv
from statistics import mean,median,pstdev

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def run_test_loops(model_path,model_type,file_name,threads,f_loop,f_stats,roop):

    interpreter = Interpreter(model_path=model_path, num_threads=threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    
    per_loop_run_tme = 0.0
    total_run_time = 0.0
    result = None
    input_shape = input_details[0]['shape']
    
    if (model_path.find('full_interger') != -1 or model_path.find('full_int')!= -1 or model_path.find('detection_default')!= -1):
        if(model_type == "YOLOv5-Lite" or model_type == "EfficientDet_Lite"):
            data_type = np.uint8
        else:
            data_type = np.int8
    else:
        data_type = np.float32 
         
    input_data = np.array(np.random.random_sample(input_shape), dtype=data_type)

    run_times = []

    for i in range(roop):
        s = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]['index'])
        per_loop_run_tme = (time.time() - s)
        total_run_time += per_loop_run_tme
        print("Model : "+str(model_type)+" "+str(file_name)+" Loop : "+str(i)+" run time : "+str(per_loop_run_tme*1000)+" ms")
        run_time = int(per_loop_run_tme*1000)
        run_times.append(run_time)
        row=str(model_type)+","+str(file_name)+","+str(input_shape)+","+str(threads)+","+str(run_time)+"\n"
        f_loop.write(row)

    row=str(model_type)+","+str(file_name)+","+str(input_shape)+","+str(threads)+","+str(mean(run_times))+","+str(median(run_times))+","+str(pstdev(run_times))+"\n"
    f_stats.write(row)

    print(f'Average run time: {total_run_time/roop*1000}ms')

if __name__=="__main__":

    model_files_path = getListOfFiles("model_files")
    max_threads = 4
    roop = 500

    f_loop = open('runtimes.csv', 'w')
    f_stats = open('runtime_stats.csv', 'w')
    
    row="Model Type,Model Name,Input Size,Threads,Loop,RunTime\n"
    f_loop.write(row)

    row="Model Type,Model Name,Input Size,Threads,Mean,Median,Std Dev\n"
    f_stats.write(row)

    '''
    model_path = "model_files/EfficientDet/03_integer_quantization/efficientdet_d0_416x416_integer_quant.tflite"
    file_name_splits = model_path.split('/')
    file_name = file_name_splits[len(file_name_splits)-1]
    model_type = file_name_splits[1]
    print("Running simulation for : "+str(model_type)+" with file : "+str(file_name))
    print("Using file : "+str(model_path))
    run_test_loops(model_path,model_type,file_name)
    '''
    for model_path in model_files_path:
        for thds in range(max_threads):
            file_name_splits = model_path.split('/')
            file_name = file_name_splits[len(file_name_splits)-1]
            model_type = file_name_splits[1]
            print("Running simulation for : "+str(model_type)+" with file : "+str(file_name))
            print("Using file : "+str(model_path))
            run_test_loops(model_path,model_type,file_name,(thds+1),f_loop,f_stats,roop)
    
    f_loop.close()
    f_stats.close()
    