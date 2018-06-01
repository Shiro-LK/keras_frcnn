import cv2
import numpy as np

def get_data(input_path, channels=4, path=''):
    '''
        we consider that one channel is an image like that (W, H, 1)
        WARNING : this function take in account until 6 channels, modified it if you need more channels. Moreover, the filename of each image have to be : 'basename_channelimage.png' 
        example with 2 channels : image_red.png, image_bue.png, width, height, x1, y1, x2, y2, class_name, imageset
    '''
    found_bg = False
    all_imgs = {}

    classes_count = {}
    classes_count_train = {}
    classes_count_test = {}

    class_mapping = {}

    visualise = True
    
    with open(input_path,'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            if channels == 2:
                (filename1,filename2,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filenames = [filename1, filename2]
            elif channels == 4:
                (filename1,filename2,filename3,filename4,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filenames = [filename1,filename2,filename3,filename4]
            elif channels ==5:
                (filename1,filename2,filename3,filename4,filename5,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filenames = [filename1,filename2,filename3,filename4,filename5]
            elif channels == 6:
                (filename1,filename2,filename3,filename4,filename5,filename6,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filenames = [filename1,filename2,filename3,filename4,filename5,filename6]
            
            basename = filenames[0][:filenames[0].rfind('_')+1]
            if imageset not in ('training', 'testing'):
                print('Imageset of ' + basename + 'is neither training nor validation, skipping image')
                continue
                
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if imageset == 'training':
                if class_name not in classes_count_train:
                    classes_count_train[class_name] = 1
                else:
                    classes_count_train[class_name] += 1 
            elif imageset == 'testing':
                if class_name not in classes_count_test:
                    classes_count_test[class_name] = 1
                else:
                    classes_count_test[class_name] += 1
                    
                    
            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if basename not in all_imgs:
                all_imgs[basename] = {}
                #print(filename)
                #img = cv2.imread(filename)
                #(rows,cols) = img.shape[:2]
                all_imgs[basename]['filepath'] = [path+filename for filename in filenames]
                all_imgs[basename]['width'] = width
                all_imgs[basename]['height'] = height
                all_imgs[basename]['bboxes'] = []
                all_imgs[basename]['imageset'] = imageset
            
            all_imgs[basename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])
        
        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch
        
        if len(classes_count_train.keys()) == len(classes_count_test.keys()):
            print('Number of classes : ', len(classes_count_train.keys()))
            return all_data, classes_count, class_mapping, classes_count_train, classes_count_test
        else:
            print('Number of classes train: ', len(classes_count_train.keys()))
            print(classes_count_train)
            print('Number of classes test: ', len(classes_count_test.keys()))
            print(classes_count_test)
            return None, None, None, None, None


