#include "od_cuda.hpp"

pthread_mutex_t mtx_od_debug_cuda;

//-------------------------------------------------------------------------------------------------------------------------------//

std::string class_names_file = "./models/yolov4-tiny-presage.names";
std::string cfg_file_name = "./models/yolov4-tiny-presage.cfg";
std::string weights_file_name = "./models/yolov4-tiny-presage.weights";

std::vector<std::string> class_names;
cv::dnn::Net *net;
//std::vector< std::string > output_names;
//std::vector<cv::Mat> detections;

std::vector<int> indices;
std::vector<cv::Rect> boxes;
std::vector<float> scores;
std::vector<cv::Point> centroids;
std::vector<double> obj_distances;

cv::Point bed_center;
double proximity_detection_threshold = 200;
int close_persons  = 0;
int far_persons = 0;

cv::Mat debug_image;

//-------------------------------------------------------------------------------------------------------------------------------//

std::vector<cv::Mat> run_inference(cv::Mat frame)
{
    std::vector<cv::Mat> detections;
    cv::Mat blob;
    std::vector< std::string > output_names;

    output_names = net->getUnconnectedOutLayersNames();
    std::cout<<"Read output names "<<std::endl;

    std::cout<<"Read input frame, creating blob "<<std::endl;
    cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(288, 288), cv::Scalar(), true, false, CV_32F);
    std::cout<<"Created blob, setting input "<<std::endl;
    net->setInput(blob);
    std::cout<<"Set input blob "<<std::endl;

    std::cout<<"Starting inference "<<std::endl;
    net->forward(detections, output_names);
    std::cout<<"Finished inference "<<std::endl;
    return detections;

}


void parse_yolo_output( cv::Mat frame,std::vector<cv::Mat> detections)
{
    std::cout<<"Finding bbox vertices and confidence "<<std::endl;
        for (auto& output : detections)
        {
            const auto num_boxes = output.rows;
            for (int i = 0; i < num_boxes; i++)
            {
                auto x = output.at<float>(i, 0) * frame.cols;
                auto y = output.at<float>(i, 1) * frame.rows;
                auto width = output.at<float>(i, 2) * frame.cols;
                auto height = output.at<float>(i, 3) * frame.rows;
                cv::Rect rect(x - width/2, y - height/2, width, height);

                auto confidence = *output.ptr<float>(i, 5);
                if (confidence >= CONFIDENCE_THRESHOLD)
                    {
                        boxes.push_back(rect);
                        scores.push_back(confidence);
                    }
            }
        }
        
        std::cout<<"running NMS on boxes "<<std::endl;
        cv::dnn::NMSBoxes(boxes, scores, 0.0, NMS_THRESHOLD, indices);
            
}

void init_model()
{
    pthread_mutex_init(&mtx_od_debug_cuda, NULL);

    std::cout<<"Loading class names "<<std::endl;
    
    {
        std::ifstream class_file("./models/yolov4-tiny-presage.names");
        if (!class_file)
        {
            std::cerr << "failed to open classes.txt\n";
            exit(-1);
        }

        std::string line;
        while (std::getline(class_file, line))
            class_names.push_back(line);
    }
    
    //class_names.push_back("person");
    std::cout<<"Loaded class names "<<std::endl;

    std::cout<<"Loading weight files "<<std::endl; 
    net = new cv::dnn::Net();
    *(net) = cv::dnn::readNetFromDarknet("./models/yolov4-tiny-presage.cfg",
                                        "./models/yolov4-tiny-presage.weights");
    std::cout<<"Loaded weight files "<<std::endl; 

    std::cout<<"Setting DNN perfs "<<std::endl;
    net->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    // net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout<<"Set up DNN perfs "<<std::endl;

    
}

void od_gen_debug_image(cv::Mat frame, std::string stats)
{   

    std::cout<<"Locking mtx_od_debug_cuda in od_gen_debug_image "<<std::endl;
    std::cout << "Size in od_gen_debug_image " << frame.size << std::endl;
    pthread_mutex_lock(&mtx_od_debug_cuda);
    std::cout<<"Lock on mtx_od_debug_cuda od_gen_debug_image"<<std::endl;
    debug_image = frame.clone();
    

    std::cout<<"drawing boxes on the frame for objects : "<<indices.size()<<" with stats : "<<stats.c_str()<<std::endl;
    for (size_t i = 0; i < indices.size(); ++i)
            
            {
                cv::Scalar color = {255, 0, 0};
                auto idx = indices[i];
                const auto& rect = boxes[idx];
                cv::rectangle(debug_image, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 1.8);

                std::ostringstream label_ss;
                //label_ss << class_names[0] << ": " << std::fixed << std::setprecision(2) << scores[idx];
                label_ss << "person : " << std::fixed << std::setprecision(2) << scores[idx];
                std::cout<<"idx : "<<idx<<" person : " << std::fixed << std::setprecision(2) << scores[idx] << std::endl;
                auto label = label_ss.str();
                cv::putText(debug_image, label.c_str(), cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 0, 0),2.0);

                if(obj_distances[i]<proximity_detection_threshold)
                    cv::line(debug_image, bed_center, centroids[i], cv::Scalar(0, 255, 0),2);
                else
                    cv::line(debug_image, bed_center, centroids[i], cv::Scalar(0, 0, 255), 1);

            }

    std::cout<<"Finished drawing boxes on the frame for objects : "<<indices.size()<<" with stats : "<<stats.c_str()<<std::endl;

    cv::putText(debug_image, stats.c_str(), cv::Point(0, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 255, 0),2.0);

    std::ostringstream p_counts_ss;
    p_counts_ss << std::fixed << std::setprecision(2);
    p_counts_ss << "Close : " << close_persons<<"  Far : "<<far_persons;
    std::cout<<"Close : " << close_persons<<"  Far : "<<far_persons<<std::endl;
    auto p_counts = p_counts_ss.str();
    cv::putText(debug_image, p_counts.c_str(), cv::Point(0, 25), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 255, 0),2.0);

    /*
    try
    {
        cv::imwrite("processed_image.jpg", debug_image);
    }
    catch( cv::Exception& e )
    {
        const char* err_msg = e.what();
        std::cout << "exception caught: " << err_msg << std::endl;
    }

    */
   
    std::cout<<"Unlocking mtx_od_debug_cuda od_gen_debug_image "<<std::endl;
    pthread_mutex_unlock(&mtx_od_debug_cuda);
    std::cout<<"Unlock on mtx_od_debug_cuda od_gen_debug_image "<<std::endl;

    //cv::namedWindow("output");
    //cv::imshow("output", debug_image);

    
    
}

cv::Mat get_od_frame()
{
    std::cout<<"Locking mtx_od_debug_cuda in get_od_frame "<<std::endl;
    pthread_mutex_lock(&mtx_od_debug_cuda);
    std::cout<<"Lock on mtx_od_debug_cuda in get_od_frame "<<std::endl;
	cv::Mat clone_debug_image = debug_image.clone();
	pthread_mutex_unlock(&mtx_od_debug_cuda);
	std::cout<<"Unlock mtx_od_debug_cuda in get_od_frame "<<std::endl;

	return clone_debug_image;
}

void reset_all_vectors()
{
    indices.clear();
    boxes.clear();
    scores.clear();
    centroids.clear();
    obj_distances.clear();
    close_persons = 0;
    far_persons = 0;
}


cv::Point compute2DPolygonCentroid(cv::Point* vertices, int vertexCount)
{
    cv::Point centroid = cv::Point(0, 0);
    double signedArea = 0.0;
    double x0 = 0.0; // Current vertex X
    double y0 = 0.0; // Current vertex Y
    double x1 = 0.0; // Next vertex X
    double y1 = 0.0; // Next vertex Y
    double a = 0.0;  // Partial signed area

    int lastdex = vertexCount-1;
    const cv::Point* prev = &(vertices[lastdex]);
    const cv::Point* next;

    // For all vertices in a loop
    for (int i=0; i<vertexCount; ++i)
    {
        next = &(vertices[i]);
        x0 = prev->x;
        y0 = prev->y;
        x1 = next->x;
        y1 = next->y;
        a = x0*y1 - x1*y0;
        signedArea += a;
        centroid.x += (x0 + x1)*a;
        centroid.y += (y0 + y1)*a;
        prev = next;
    }

    signedArea *= 0.5;
    centroid.x /= (6.0*signedArea);
    centroid.y /= (6.0*signedArea);

    return centroid;
}


void set_up_bed_center(int *bed_mask)
{
    cv::Point bed_points[4];
    
    int counter = 0;
    for(int i=0;i<4;i++)
    {
        bed_points[i].x = bed_mask[counter];
        bed_points[i].y = bed_mask[counter+1];
        counter+=2;
    }

    bed_center = compute2DPolygonCentroid(bed_points,4);

}

int do_proximity_detection()
{
    int num_of_close_persons = 0;
    int det_center_x;
    int det_center_y;
    float distance = 0.0;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        auto idx = indices[i];
        const auto& rect = boxes[idx];
        cv::Point center_of_rect = (rect.br() + rect.tl())*0.5;
        centroids.push_back(center_of_rect);
        distance = cv::norm(center_of_rect-bed_center);
        obj_distances.push_back(distance);

        if(distance < proximity_detection_threshold)
        {
            num_of_close_persons++;
            close_persons++;
        }
        else
        {
            far_persons++;
        }
    }

    return num_of_close_persons;
}

void run_destructor()
{
    cv::destroyAllWindows();
    delete(net);
}