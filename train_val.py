import sys
sys.path.append('/home/kh31/jmh/thundersvm/python/thundersvm')

import numpy as np
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from thundersvm import SVC

def load_npy_file(file_path, root):
    arr = np.load(file_path)
    label = 0 if '0_real' in root else 1
    return arr.squeeze(), label

def getdata(path):
    X, y = [], []
    files_to_load = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                files_to_load.append((file_path, root))

    with ProcessPoolExecutor() as executor:
        results = executor.map(load_npy_file_wrapper, files_to_load)

    for data, label in results:
        X.append(data)
        y.append(label)

    return np.array(X), np.array(y)

def load_npy_file_wrapper(args):
    return load_npy_file(*args)

def main(args):
    # 训练模型
    print("Loading training data...")
    train_X, train_y = getdata(args.train_path)
    print(f"Training data loaded. Samples: {len(train_y)}")
    
    print("Training SVM model...")
    svm_model = SVC(
        kernel=args.kernel,
        probability=args.probability,
        n_jobs=args.n_jobs,
        verbose=args.verbose
    )
    svm_model.fit(train_X, train_y)
    
    # 验证数据集
    type_list = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'stylegan2',
                 'whichfaceisreal', 'ADM', 'Glide', 'Midjourney', 'stable_diffusion_v_1_4',
                 'stable_diffusion_v_1_5', 'VQDM', 'wukong', 'DALLE2', 'SDXL', 'flux']
    
    total_acc, total_auc = 0, 0
    for method in type_list:
        test_path = os.path.join(args.test_root, method)
        print(f"Testing on {method}...")
        
        val_X, val_y = getdata(test_path)
        y_pred = svm_model.predict(val_X)
        y_prob = svm_model.predict_proba(val_X)[:, 1]
        
        accuracy = accuracy_score(val_y, y_pred)
        auc = roc_auc_score(val_y, y_prob)
        
        with open(args.result_file, 'a') as f:
            print(f"{method} Accuracy: {accuracy}", file=f)
            print(f"{method} AUC Score: {auc}", file=f)
        
        total_acc += accuracy
        total_auc += auc
    
    # 输出平均结果
    avg_acc = total_acc / len(type_list)
    avg_auc = total_auc / len(type_list)
    with open(args.result_file, 'a') as f:
        print("\nFinal Summary:", file=f)
        print(f"Average Accuracy: {avg_acc:.4f}", file=f)
        print(f"Average AUC: {avg_auc:.4f}", file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVM Training and Evaluation Script')
    
    # 必需参数
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--test_root', type=str, required=True,
                       help='Root directory containing test datasets')
    parser.add_argument('--result_file', type=str, required=True,
                       help='Path to output results file')
    
    # 可选参数
    parser.add_argument('--kernel', type=str, default='rbf',
                       choices=['linear', 'poly', 'rbf', 'sigmoid'],
                       help='SVM kernel type')
    parser.add_argument('--probability', action='store_true',
                       help='Enable probability estimates')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    
    # 清空或创建结果文件
    open(args.result_file, 'w').close()
    
    main(args)