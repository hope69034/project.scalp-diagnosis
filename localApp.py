# from, import
from flask import Flask, render_template, request
# 웹- 업로드 시 실행하지 않게 미리 임폴트
import torchvision
from torchvision import transforms
import os
from torch.utils.data import Dataset,DataLoader
import torch

# 웹페이지에서 이미지를 업로드 받은 후 연산시간을 줄이기 위해 미리 로드하는 것
# PATH
PATH1 = './content/scalp_weights/'+'aram_model1.pt'  # 모델1
PATH2 = './content/scalp_weights/'+'aram_model2.pt'  # 모델2
PATH3 = './content/scalp_weights/'+'aram_model3.pt'  # 모델3
PATH4 = './content/scalp_weights/'+'aram_model4.pt'  # 모델4
PATH5 = './content/scalp_weights/'+'aram_model5.pt'  # 모델5
PATH6 = './content/scalp_weights/'+'aram_model6.pt'  # 모델6
# cuda        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model load        
model1 = torch.load(PATH1, map_location=device)
model2 = torch.load(PATH2, map_location=device)
model3 = torch.load(PATH3, map_location=device)
model4 = torch.load(PATH4, map_location=device)
model5 = torch.load(PATH5, map_location=device)
model6 = torch.load(PATH6, map_location=device)

# 모델 평가모드로 전환 # 평가모드와 학습모드의 layer 구성이 다르다
model1.eval() 
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()

## Web Server Code
app = Flask(__name__, static_folder='./content/static', # 스택틱폴더, 스택틱템플릿폴더 지정
                    template_folder='./content/templates')

@app.route('/')
def home():
    menu = {'home':1, 'menu':0}
    return render_template('index.html', menu=menu, )     

@app.route('/menu', methods=['GET','POST'])
def menu():
    menu = {'home':0, 'menu':1}
    if request.method == 'GET':
        languages = []
        return render_template('menu.html', menu=menu, 
                                options=languages) # 서버에서 클라이언트로 정보 전달
    else:
        f_image = request.files['image']
        #fname = f_image.filename     # 사용자가 업로드한 파일의 파일 이름
        newname = 'upload.jpg' # 업로드폴더를리셋하기위해일정한새이름을지정 upload.jpg로 업로드
        filename = os.path.join(app.static_folder, 'upload/upload/') + newname  # 유저가 업로드한 사진이 저장되는 공간애 newname의 이름으로 저장 
            #여기서 저장경로는 스택틱폴더경로/업로드/업로드
        f_image.save(filename) #파일세이브
        return render_template('menu_spinner.html', menu=menu ,  filename=filename )
        
@app.route('/menu_res', methods=['POST'])
def menu_res():
    menu = {'home':0, 'menu':1}
#########################################################################
## model.test code
# test 이미지파일 전처리, 텐서화
    # 전처리-트랜스폼 규칙 선언 # model1_train 코드의 validation set 의 트랜스폼 규칙과 동일하게 
    transforms_test = transforms.Compose([  transforms.Resize([int(600), int(600)], interpolation=transforms.InterpolationMode.BOX),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    ])
    # root 경로 폴더 속 jpg를 전처리, 텐서화 (rood 속에 폴더를 하나 더 만들어서 jpg를 묶어야 함)
    testset = torchvision.datasets.ImageFolder(root = './content/static/upload' ,
                        transform = transforms_test)
# DataLoader를 통해 네트워크에 올리기 
    testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0)
    with torch.no_grad(): # 평가할 땐  gradient를 backpropagation 하지 않기 때문에 no grad로 gradient 계산을 막아서 연산 속도를 높인다
        for data, target in testloader:                                   
            data, target  = data.to(device), target.to(device) 
            output1 = model1(data)   # model1에 데이터를 넣어서 아웃풋 > [a,b,c,d] 각 0,1,2,3 의 확률값 리턴 가장 큰 것이 pred
            output2 = model2(data) 
            output3 = model3(data) 
            output4 = model4(data) 
            output5 = model5(data) 
            output6 = model6(data)  
# predict # # 0~3값만 뽑기 
    m1p = output1.argmax(dim=1, keepdim=True)[0][0].tolist()
    m2p = output2.argmax(dim=1, keepdim=True)[0][0].tolist()
    m3p = output3.argmax(dim=1, keepdim=True)[0][0].tolist()
    m4p = output4.argmax(dim=1, keepdim=True)[0][0].tolist()
    m5p = output5.argmax(dim=1, keepdim=True)[0][0].tolist()
    m6p = output6.argmax(dim=1, keepdim=True)[0][0].tolist()
# 진단
    d_list = [] # 두피유형진단결과
    # 두피 유형 진단법
    if m1p <= 1.5 and m2p <= 1.5 and m3p <= 1.5 and m4p <= 1.5 and m5p <= 1.5 and m6p <= 1.5 :
        d1 = '정상입니다.'
        d_list.append(d1)
    elif m1p > 1.5 and m2p <=1.5 and m3p <=1.5 and m4p <=1.5 and m5p <=1.5 and m6p <=1.5 :
        d2 = '건성 두피입니다.'
        d_list.append(d2)
    elif m1p <=1.5 and m2p > 1.5 and m3p <=1.5 and m4p <=1.5 and m5p <=1.5 and m6p <=1.5 :
        d3 = '지성 두피입니다.'
        d_list.append(d3)
    elif m2p <=1.5 and m3p > 1.5 and m4p <=1.5 and m5p <=1.5 and m6p <=1.5 :
        d4 = '민감성 두피입니다.'
        d_list.append(d4)
    elif m2p > 1.5 and m3p > 1.5 and m4p <=1.5 and m6p <=1.5 :
        d5 = '지루성 두피입니다.'
        d_list.append(d5)
    elif m3p <=1.5 and m4p > 1.5 and m6p <=1.5 :
        d6 = '염증성 두피입니다.'
        d_list.append(d6)
    elif m3p <=1.5 and m4p <=1.5 and m5p > 1.5 and m6p <=1.5 :
        d7 = '비듬성 두피입니다.'
        d_list.append(d7)
    elif m1p <=1.5 and m2p > 1.5 and m3p <=1.5 and m4p <=1.5 and m5p <=1.5 and m6p > 1.5 :
        d8 = '탈모입니다.'
        d_list.append(d8)
    else:
        d9 = '복합성 두피입니다.'
        d_list.append(d9)
#########################################################################
## Web Server Code
# 모델 실행후 결과를 돌려줌
    final = d_list[0] # 두피유형판단
    result = {'미세각질':m1p, '피지과다':m2p,'모낭사이홍반':m3p,'모낭홍반농포':m4p,'비듬':m5p,'탈모':m6p} # result=result
    final2 = '0:양호, 1:경증, 2:중등도, 3:중증' 
    filename = request.args['filename']  #html에서 넘겨준 파일네임을 리퀘스트로 불러오기
    mtime = int(os.stat(filename).st_mtime) # 업로드한 시간값불러오기 > 큐변경 > 화면갱신 4
    return render_template('menu_res.html',  final2=final2,final=final, menu=menu, result=result, 
                            m1p=m1p,m2p=m2p,m3p=m3p,m4p=m4p,m5p=m5p,m6p=m6p , mtime=mtime    )
# solution 화면            
@app.route('/menu_res_inf')
def menu_res_inf(): 
    menu = {'home':0, 'menu':0}
    m1p = int(request.args['m1p']) # menu_res에서 m1p 를 가져오기
    m2p = int(request.args['m2p'])
    m3p = int(request.args['m3p'])
    m4p = int(request.args['m4p'])
    m5p = int(request.args['m5p'])
    m6p = int(request.args['m6p'])
    return render_template('menu_res_inf.html', menu=menu, m1p=m1p,m2p=m2p,m3p=m3p,m4p=m4p,m5p=m5p,m6p=m6p, )
if __name__ == '__main__':
    app.run()