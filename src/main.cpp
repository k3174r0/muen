#include "darknet.h"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "opencv/highgui.h"
#include <iostream>
#include <math.h>
// #include <omp.h>

float abs_mean(float *x, int n)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i){
        sum += fabs(x[i]);
    }
    return sum/n;
}

void calculate_loss(float *output, float *delta, int n, float thresh)
{
    int i;
    float mean = mean_array(output, n); 
    float var = variance_array(output, n);
    for(i = 0; i < n; ++i){
        if(delta[i] > mean + thresh*sqrt(var)) delta[i] = output[i];
        else delta[i] = 0;
    }
}

void optimize_picture(network *net, image orig, int max_layer, float scale, float rate, float thresh, int norm)
{
    //scale_image(orig, 2);
    //translate_image(orig, -1);
    net->n = max_layer + 1;

    int dx = rand()%16 - 8;
    int dy = rand()%16 - 8;
    int flip = rand()%2;

    image crop = crop_image(orig, dx, dy, orig.w, orig.h);
    image im = resize_image(crop, (int)(orig.w * scale), (int)(orig.h * scale));
    if(flip) flip_image(im);

    resize_network(net, im.w, im.h);
    layer last = net->layers[net->n-1];
    //net->layers[net->n - 1].activation = LINEAR;

    image delta = make_image(im.w, im.h, im.c);

#ifdef GPU
    net->delta_gpu = cuda_make_array(delta.data, im.w*im.h*im.c);
    copy_cpu(net->inputs, im.data, 1, net->input, 1);
    forward_network_gpu(net);
    copy_gpu(last.outputs, last.output_gpu, 1, last.delta_gpu, 1);

    cuda_pull_array(last.delta_gpu, last.delta, last.outputs);
    calculate_loss(last.delta, last.delta, last.outputs, thresh);
    cuda_push_array(last.delta_gpu, last.delta, last.outputs);

    backward_network_gpu(net);

    cuda_pull_array(net->delta_gpu, delta.data, im.w*im.h*im.c);
    cuda_free(net->delta_gpu);
    net->delta_gpu = 0;
#else
    printf("\nnet: %d %d %d im: %d %d %d\n", net->w, net->h, net->inputs, im.w, im.h, im.c);
    copy_cpu(net->inputs, im.data, 1, net->input, 1);
    net->delta = delta.data;
    forward_network(net);
    copy_cpu(last.outputs, last.output, 1, last.delta, 1);
    calculate_loss(last.output, last.delta, last.outputs, thresh);
    backward_network(net);
#endif

    if(flip) flip_image(delta);
    //normalize_array(delta.data, delta.w*delta.h*delta.c);
    image resized = resize_image(delta, orig.w, orig.h);
    image out = crop_image(resized, -dx, -dy, orig.w, orig.h);

    image gray = make_image(out.w, out.h, out.c);
    fill_image(gray, .5);
    axpy_cpu(orig.w*orig.h*orig.c, -1, orig.data, 1, gray.data, 1);
    axpy_cpu(orig.w*orig.h*orig.c, .1, gray.data, 1, out.data, 1);

    if(norm) normalize_array(out.data, out.w*out.h*out.c);
    axpy_cpu(orig.w*orig.h*orig.c, rate, out.data, 1, orig.data, 1);

    constrain_image(orig);

    free_image(crop);
    free_image(im);
    free_image(delta);
    free_image(resized);
    free_image(out);

}

void smooth(image recon, image update, float lambda, int num)
{
    int i, j, k;
    int ii, jj;
    for(k = 0; k < recon.c; ++k){
        for(j = 0; j < recon.h; ++j){
            for(i = 0; i < recon.w; ++i){
                int out_index = i + recon.w*(j + recon.h*k);
                for(jj = j-num; jj <= j + num && jj < recon.h; ++jj){
                    if (jj < 0) continue;
                    for(ii = i-num; ii <= i + num && ii < recon.w; ++ii){
                        if (ii < 0) continue;
                        int in_index = ii + recon.w*(jj + recon.h*k);
                        update.data[out_index] += lambda * (recon.data[in_index] - recon.data[out_index]);
                    }
                }
            }
        }
    }
}

void reconstruct_picture(network *net, float *features, image recon, image update, float rate, float momentum, float lambda, int smooth_size, int iters)
{
    int iter = 0;
    for (iter = 0; iter < iters; ++iter) {
        image delta = make_image(recon.w, recon.h, recon.c);

#ifdef GPU
        layer l = get_network_output_layer(net);
        cuda_push_array(net->input_gpu, recon.data, recon.w*recon.h*recon.c);
        //cuda_push_array(net->truth_gpu, features, net->truths);
        net->delta_gpu = cuda_make_array(delta.data, delta.w*delta.h*delta.c);

        forward_network_gpu(net);
        cuda_push_array(l.delta_gpu, features, l.outputs);
        axpy_gpu(l.outputs, -1, l.output_gpu, 1, l.delta_gpu, 1);
        backward_network_gpu(net);

        cuda_pull_array(net->delta_gpu, delta.data, delta.w*delta.h*delta.c);

        cuda_free(net->delta_gpu);
#else
        net->input = recon.data;
        net->delta = delta.data;
        net->truth = features;

        forward_network(net);
        backward_network(net);
#endif

        //normalize_array(delta.data, delta.w*delta.h*delta.c);
        axpy_cpu(recon.w*recon.h*recon.c, 1, delta.data, 1, update.data, 1);
        //smooth(recon, update, lambda, smooth_size);

        axpy_cpu(recon.w*recon.h*recon.c, rate, update.data, 1, recon.data, 1);
        scal_cpu(recon.w*recon.h*recon.c, momentum, update.data, 1);

        float mag = mag_array(delta.data, recon.w*recon.h*recon.c);
        printf("mag: %f\n", mag);
        //scal_cpu(recon.w*recon.h*recon.c, 600/mag, recon.data, 1);

        constrain_image(recon);
        free_image(delta);
    }
}

int main(int argc, char **argv)
{
    if(argc < 2){
        std::cerr << "Usage: ./muen (int)layer (int)iteration" << std::endl;
        return 1;
    }

    static char *cfg = const_cast<char *>("../data/jnet-conv.cfg");
    static char *weights = const_cast<char *>("../data/jnet-conv.weights");

    // char *input = argv[1];
    int max_layer = atoi(argv[1]);
    int iters = atoi(argv[2]);

    // int range = find_int_arg(argc, argv, "-range", 1);
    // int norm = find_int_arg(argc, argv, "-norm", 1);
    // int rounds = find_int_arg(argc, argv, "-rounds", 1);
    // int iters = find_int_arg(argc, argv, "-iters", 10);
    // int octaves = find_int_arg(argc, argv, "-octaves", 4);
    // float zoom = find_float_arg(argc, argv, "-zoom", 1.);
    // float rate = find_float_arg(argc, argv, "-rate", .04);
    // float thresh = find_float_arg(argc, argv, "-thresh", 1.);
    // float rotate = find_float_arg(argc, argv, "-rotate", 0);
    // float momentum = find_float_arg(argc, argv, "-momentum", .9);
    // float lambda = find_float_arg(argc, argv, "-lambda", .01);
    // char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    // int reconstruct = find_arg(argc, argv, "-reconstruct");
    // int smooth_size = find_int_arg(argc, argv, "-smooth", 1);

    int range = 1;
    int norm = 1;
    int rounds = 1;
    // int iters = 10;
    int octaves = 4;
    float zoom = 1.;
    float rate = .04;
    float thresh = 1.;
    float rotate = 0;
    float momentum = .9;
    float lambda = .01;
    char *prefix = 0;
    int reconstruct = 0;
    int smooth_size = 1;

    network *net = load_network(cfg, weights, 0);

    set_batch_network(net, 1);

    float *features = 0;
    image update;
 
    cv::VideoCapture cap(0);
    // cv::VideoCapture cap("http://admin:@192.168.10.1/media/?action=stream.mjpeg");
    cap.set(3, 480);//width
    cap.set(4, 360);//height

    cv::namedWindow("image", CV_WINDOW_NORMAL);
    // cv::setWindowProperty("image", cv::WND_PROP_FULLSCREEN,CV_WINDOW_FULLSCREEN);
    while(1){
        cv::Mat im1, im2;
        cap >> im1;

        // cv::resize(im1, im1, cv::Size(), 0.5, 0.5);

        //Mat->image
        //Reference : https://qiita.com/TaroYamada/items/5a070271151383541622
        im1.convertTo(im2, CV_32FC3, 1.0/255);
        std::vector<cv::Mat> tmp;
        cv::split(im2, tmp);
        int size = im2.size().width * im2.size().height;
        int fsize = size * sizeof(float);
        image im3 = make_image(im2.size().width, im2.size().height, 3);
        float*p = im3.data;
        memcpy((unsigned char*)p, tmp[2].data, fsize);
        p+= size;
        memcpy((unsigned char*)p, tmp[1].data, fsize);
        p+= size; 
        memcpy((unsigned char*)p, tmp[0].data, fsize);

        int e;
        int n;
        for(e = 0; e < rounds; ++e){
            for(n = 0; n < iters; ++n){  
                int layer = max_layer + rand()%range - range/2;
                int octave = rand()%octaves;
                optimize_picture(net, im3, layer, 1/pow(1.33333333, octave), rate, thresh, norm);
            }
        }
        show_image(im3,"image", 1);
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}