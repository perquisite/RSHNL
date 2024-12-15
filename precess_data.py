import scipy.io as sio

data_name = 'xmedia'


if data_name == 'xmedia':
        valid_len = 500
        path = 'datasets/XMediaFeatures.mat'
        all_data = sio.loadmat(path)
        img_test = all_data['I_te_CNN'].astype('float32')   # Features of test set for image data, CNN feature
        img_train = all_data['I_tr_CNN'].astype('float32')   # Features of training set for image data, CNN feature
        text_test = all_data['T_te_BOW'].astype('float32')   # Features of test set for text data, BOW feature
        text_train = all_data['T_tr_BOW'].astype('float32')   # Features of training set for text data, BOW feature

        label_test_img = all_data['teImgCat'].reshape([-1]).astype('float16') # category label of test set for image data
        label_train_img = all_data['trImgCat'].reshape([-1]).astype('float16') # category label of training set for image data

sio.savemat('XMediaFeatures.mat',{'I_te_CNN': img_test,
                                  'I_tr_CNN': img_train,
                                  'T_te_BOW':text_test,
                                  'T_tr_BOW':text_train,
                                  'teImgCat':label_test_img,
                                  'trImgCat':label_train_img}
                                  )