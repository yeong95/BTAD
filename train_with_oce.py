from sklearn.random_projection import SparseRandomProjection
from sampling_methods.kcenter_greedy import kCenterGreedy
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve, auc
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import argparse
import shutil
import faiss
import torch
import glob
import cv2
import os
import time
import wandb

from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import pickle
from sampling_methods.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from resnet import wide_resnet50_2, resnet34
from de_resnet import de_wide_resnet50_2
import random

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist


class NN():

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):


        # dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        dist = torch.cdist(x, self.train_pts, self.p)

        knn = dist.topk(self.k, largest=False)


        return knn

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)

def prep_dirs(root):
    # make embeddings dir
    # embeddings_path = os.path.join(root, 'embeddings')
    embeddings_path = os.path.join('./', 'embeddings', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    # os.makedirs(source_code_save_path, exist_ok=True)
    # copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README','samples','LICENSE', 'Notebook']) # copy source code
    return embeddings_path, sample_path, source_code_save_path

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def reshape_embedding(embedding):    # feature vector 펼쳐서 embedding_list에 추가 
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list


#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    def __init__(self, root, transform, phase, args):
        if phase=='train':
            self.img_path = os.path.join(root, f'{args.train_folder}')
        else:
            self.img_path = os.path.join(root, f'{args.test_folder}')
        self.transform = transform
        # load dataset
        self.img_paths,  self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
 
        img_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        if args.category == '코팅부':
            normal_names = ['코팅부 경계부 불량', '코팅부 미코팅', '코팅부 접힘', '코팅부 줄무늬', '코팅부 코팅불량', 'augmentation']
        elif args.category == '무지부':
            normal_names = ['무지부 줄무늬', 'augmentation']
        elif args.category == '무지부_코팅부':
            normal_names = ['무지부 줄무늬', '무지부_aug', '코팅부 경계부 불량', '코팅부 미코팅', '코팅부 접힘', '코팅부 줄무늬', '코팅부 코팅불량', '코팅부_aug']
        
        print("normal class: ", normal_names)
        
        for defect_type in defect_types:
            if defect_type in normal_names:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))
        
        print("total data: ", len(img_tot_paths))
        return img_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label, os.path.basename(img_path[:-4]), img_type


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    


def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive')
    print(false_p)
    print('false negative')
    print(false_n)
    

class STPM(pl.LightningModule):
    def __init__(self, hparams):
        super(STPM, self).__init__()

        self.save_hyperparameters(hparams)

        # self.faiss_path = args.faiss_path

        self.init_features()
        # def hook_t(module, input, output):
        #     self.features.append(output)

        self.encoder, self.model = wide_resnet50_2(pretrained=True)  
        
        if args.category == '코팅부':
            ckpt_path = 'embeddings/코팅부/wres50_코팅부.pth' 
        elif args.category == '무지부':
            ckpt_path = 'embeddings/무지부/wres50_무지부epochs19.pth' 
        elif args.category == '무지부_코팅부':
            # ckpt_path = 'embeddings/무지부_코팅부/wres50_무지부_코팅부epochs80.pth' 
            artifact = run.use_artifact('yeong/RD4AD/model:v39', type='model')
            artifact_dir = artifact.download()
            model_name = f'wres50_{args.category}'+'epochs200'+'_wandb'+'.pth'
            
        print()
        print("reverse distillation ckpt load from: ", ckpt_path)    
            
        # self.model.load_state_dict(torch.load(os.path.join(artifact_dir, model_name), map_location='cuda:0')['bn'])
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cuda:0')['bn'])


        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False

        # self.model.layer2[-1].register_forward_hook(hook_t)
        # self.model.layer3[-1].register_forward_hook(hook_t)
        # self.model.layer4[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()

        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        # transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

        self.train_class = []
        self.train_img_path = []
        # self.test_embedding_list = []
        self.nn_indices = []

        self.test_embedding_list = []

    def init_results_list(self):
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        
        self.x_type_list = []

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        # self.init_features()
        out = self.encoder(x_t)
        output = self.model(out)
        return output   # original: self.features, output

    def save_anomaly_map(self, anomaly_map, input_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)

    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms,  phase='train', args=args)
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=False, num_workers=0) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, phase='test', args=args)
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.encoder.eval()
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.version_name = self.source_code_save_path.split('/')[-2]
        self.embedding_list = []
    
    def on_test_start(self):
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.version_name = self.source_code_save_path.split('/')[-2]
        if args.feature_level == 'patch':
            self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'patch_index.faiss'))
        else:
            self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'image_index.faiss'))
            # artifact = run.use_artifact(f'yeong/PatchCore/image_index_faiss:v43', type='pickle')  # image_index_faiss:v18 is best for coating, v46: L2
            # artifact_dir = artifact.download()
            # self.index = faiss.read_index(os.path.join(artifact_dir, f'image_index.faiss'))

        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)  # second argumen: gpu device 
        self.init_results_list()
        self.start = time.time()
        
        self.score_images_list = []
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, file_name, x_type = batch  # x: (32,3, 224, 224)
        features = self(x)   # output: (32, 1000)

        if args.feature_level == 'patch':
            embeddings = []
            for feature in features:   # features[0]: (32, 512, 28, 28), features[1]: (32, 1024, 14, 14)
                m = torch.nn.AvgPool2d(3, 1, 1)
                embeddings.append(m(feature))
            embedding = embedding_concat(embeddings[0], embeddings[1])  # embedding: (28, 1536, 28, 28)
            self.embedding_list.extend(reshape_embedding(np.array(embedding)))   # embedding_list에 모든 feature vector 담겨 있음 

        elif args.feature_level == 'image':            
            # oce embedding  
            embedding = F.adaptive_max_pool2d(features, (1,1))
            self.embedding_list.extend(torch.squeeze(embedding).cpu().detach().numpy())
    
        self.train_class += list(file_name)
        self.train_img_path += list(x_type)

    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        # save
        # np.save("embeddings/무지부_코팅부/train_file_name.npy", np.array(self.train_class))
        # np.save("embeddings/무지부_코팅부/train_class.npy", np.array(self.train_img_path))
        # np.save("embeddings/무지부_코팅부/oce embedding.npy", total_embeddings)

        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings,0,0)
        print('initial embedding size : ', total_embeddings.shape)
        if args.feature_level == 'patch':
            selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*args.coreset_sampling_ratio))
            self.embedding_coreset = total_embeddings[selected_idx]
            print('final embedding size : ', self.embedding_coreset.shape)
        else:
            self.embedding_coreset = total_embeddings 
            print('final embedding size : ', self.embedding_coreset.shape)

        #faiss
        if args.distance == 'L2':
            self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])            
        elif args.distance == 'cosine':
            self.index = faiss.IndexFlatIP(self.embedding_coreset.shape[1])
            faiss.normalize_L2(self.embedding_coreset)
        
        self.index.add(self.embedding_coreset)
        faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,f'{args.feature_level}_index.faiss'))

        artifact = wandb.Artifact(f'{args.feature_level}_index_faiss', type='pickle')
        artifact.add_file(os.path.join(self.embedding_dir_path, f'{args.feature_level}_index.faiss'))
        wandb.log_artifact(artifact)
        # wandb.join()

    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        x, label, file_name, x_type = batch   # x: (1, 3, 224, 224)
        # extract embedding
        features = self(x)

        if args.feature_level == 'patch':
            embeddings = []
            for feature in features:
                m = torch.nn.AvgPool2d(3, 1, 1)
                embeddings.append(m(feature))
            embedding_ = embedding_concat(embeddings[0], embeddings[1])
            embedding_test = np.array(reshape_embedding(np.array(embedding_)))    # embedding_test: (784, 1536)  784 = 하나의 이미지 당 28*28개 만큼의 patch가 나옴, 1536 = embedding size
        else:
            # embedding_test = output

            # oce embedding
            embedding_test = F.adaptive_max_pool2d(features, (1,1))

        if args.feature_level == 'patch':
            score_patches, _ = self.index.search(embedding_test , k=args.n_neighbors)
            anomaly_map = score_patches[:,0].reshape((28,28))   # 가장 거리가 가까운 score 가져와서 anomaly map 구성. anomaly_map: (28, 28)
            N_b = score_patches[np.argmax(score_patches[:,0])]  # 각 patch별로 거리가 가장 가까운 것 중 가장 먼 patch의 nearest 9 score 
            w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
            score = w*max(score_patches[:,0]) # Image-level score

            anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
            anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)

            self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        else:
            if args.distance == 'L2':
                score_images, indices = self.index.search(embedding_test.cpu().detach().numpy().reshape(1,-1), k=args.n_neighbors)
                min_score = min(score_images[:,0])                 
                score_images /= 10**(len(str(min_score))-1)
                # score_images /= 1000
                w = (1 - (np.min(np.exp(score_images))/np.sum(np.exp(score_images))))
                score = w*min_score    
            elif args.distance == 'cosine':
                query = embedding_test.cpu().detach().numpy().reshape(1,-1)
                faiss.normalize_L2(query)                
                score_images, indices = self.index.search(query, k=args.n_neighbors)    
                w = (1 - (np.max(np.exp(score_images))/np.sum(np.exp(score_images))))
                score = w*(score_images[:,0]) # minus for cosine sim.
                score = 1- score                         

            # nn only 1 images
            # if args.distance == 'L2':
            #     score = np.squeeze(score_images[0][0])
            # else:
            #     score = 1-np.squeeze(score_images[0][0])

        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        self.x_type_list.extend(x_type)

        self.nn_indices.append(np.squeeze(indices).tolist())
        self.score_images_list.append(score_images[0])

        if args.feature_level == 'patch':
            # save images
            x = self.inv_normalize(x)
            input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
            self.save_anomaly_map(anomaly_map_resized_blur, input_x, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        # np.save(f"{self.embedding_dir_path}/test embedding pretrained.npy", self.test_embedding_list)
        # save          
        # import pdb;pdb.set_trace()
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)

        print("Total image-level prauc score :")
        precision, recall, thresholds = precision_recall_curve(self.gt_list_img_lvl, self.pred_list_img_lvl)
        auc_precision_recall = auc(recall, precision)
        print(auc_precision_recall)
        values = {'img_auc': img_auc, 'prauc': auc_precision_recall}
        self.log_dict(values)
        print('test_epoch_end')
        print("time: ", time.time() - self.start)
        
        pred_values = {'gt_list':self.gt_list_img_lvl, 'pred_list':self.pred_list_img_lvl, 'img_path_list':self.img_path_list, 'x_type':self.x_type_list, \
            'nn_indices':self.nn_indices, 'score_images': self.score_images_list}
        with open(f"{self.embedding_dir_path}/pred_values.pickle", "wb") as f:
            pickle.dump(pred_values, f)

        wandb.init(project='PatchCore')
        artifact = wandb.Artifact('result_pickle', type='pickle')
        artifact.add_file(f'{self.embedding_dir_path}/pred_values.pickle')
        run.log_artifact(artifact)
        
        run.log({'img_auc': img_auc, 'prauc': auc_precision_recall})
        # run.finish()

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'./MVTec') # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--train_folder', default='train')
    parser.add_argument('--test_folder', default='valid_test_2')
    parser.add_argument('--category', default='hazelnut')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256) # 256
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--coreset_sampling_ratio', default=0.001, type=float)
    parser.add_argument('--project_root_path', default=r'./test') # 'D:\Project_Train_Results\mvtec_anomaly_detection\210624\test') #
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    parser.add_argument('--feature_level', type=str, default='patch')
    parser.add_argument('--distance', choices=['L2', 'cosine'], default='L2')
    # parser.add_argument('--faiss_path', default=r'코팅부/lightning_logs/version_0/src/embeddings/코팅부/')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args = get_args()

    # setting wandb
    run = wandb.init(project='PatchCore')
    wandb.config.update(args)
    # args['wandb'] = wandb

    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=1, auto_select_gpus=True) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    model = STPM(hparams=args)
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)

