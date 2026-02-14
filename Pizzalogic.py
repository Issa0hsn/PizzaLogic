import numpy as np
import matplotlib.pyplot as plt
import time
# X: [dough weight, cheese weight, cheese price, peperoni weight, vegetables price, pizza price]
x = np.array([
    [210, 120, 2.80, 80, 1.50, 18.50],
    [180,  80, 1.60,  0, 0.80, 12.70],
    [250, 150, 3.90, 100, 2.20, 24.30],
    [190,  60, 1.10, 40, 0.60, 11.40],
    [230, 140, 3.20, 70, 1.80, 21.50],
    [170,  50, 0.90, 20, 0.40,  9.80],
    [280, 180, 4.50, 110, 2.70, 29.20],
    [200, 100, 1.90, 60, 1.20, 16.40],
    [240, 130, 2.90, 90, 1.90, 22.80],
    [160,  70, 1.20, 10, 0.30,  8.90],
    [270, 160, 3.70, 120, 2.50, 27.60],
    [185,  90, 1.70, 30, 0.90, 14.20],
    [260, 170, 4.10, 85, 2.40, 26.90],
    [175,  75, 1.30, 25, 0.50, 10.50],
    [290, 190, 5.00, 105, 3.00, 31.70],
    [195,  85, 1.50, 50, 1.00, 13.80],
    [220, 110, 2.30, 75, 1.70, 19.20],
    [255, 145, 3.40, 95, 2.10, 23.70],
    [165,  55, 1.00, 15, 0.20,  7.60],
    [275, 175, 4.30, 115, 2.80, 28.40],
    [205, 105, 2.10, 65, 1.40, 17.90],
    [245, 155, 3.60, 88, 2.30, 25.20],
    [172,  68, 1.15, 35, 0.70, 11.20],
    [265, 165, 3.95, 102, 2.60, 26.30],
    [188,  78, 1.40, 45, 0.85, 12.90],
    [285, 185, 4.70, 108, 2.90, 30.50],
    [192,  82, 1.55, 55, 0.95, 14.70],
    [235, 125, 2.70, 82, 1.75, 20.80],
    [208,  98, 1.80, 48, 1.10, 15.60],
    [252, 142, 3.20, 92, 2.00, 22.40],
    [178,  62, 1.05, 12, 0.35,  8.50],
    [268, 168, 4.00, 98, 2.55, 27.10],
    [202, 102, 1.95, 52, 1.25, 16.90],
    [248, 152, 3.50, 96, 2.15, 24.80],
    [162,  58, 0.95,  8, 0.25,  7.20],
    [272, 172, 4.20, 104, 2.70, 28.90],
    [183,  73, 1.25, 38, 0.65, 10.80],
    [238, 138, 3.00, 86, 1.85, 21.20],
    [212, 112, 2.35, 70, 1.55, 19.70],
    [258, 158, 3.75, 94, 2.35, 25.90],
    [168,  65, 1.10, 18, 0.45,  9.30],
    [278, 178, 4.50, 106, 2.80, 29.80],
    [198,  88, 1.60, 42, 1.05, 13.50],
    [242, 148, 3.30, 90, 2.05, 23.20],
    [215, 115, 2.40, 72, 1.60, 20.10],
    [173,  63, 1.08, 22, 0.48,  8.80],
    [262, 162, 3.80, 100, 2.45, 26.20],
    [186,  76, 1.35, 32, 0.75, 11.80],
    [246, 146, 3.40, 84, 2.10, 24.10],
    [225, 122, 2.60, 78, 1.70, 20.50]
])
# y: 0= fair ,1= expensive
y = np.array([
    0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 1, 0, 1, 0, 1, 1
])
m=x.shape[0]
w=np.array([1,1,1,1,1,1])
b=10
def prediction(x,w,b):
    f=np.dot(x,w)+b
    return 1/(1+(np.exp(-f)))
#done
def z_score(x,n):
    if n==1:
        m=1
    else:    
        m=np.mean(x)
    s = ((np.sum((x-m)**2))/n)**(1/2)
    return (x-m)/s
def unscale(origin,current,n):
    if n==1:
        m=1
    else:
        m=np.mean(origin)
    s= ((np.sum((origin-m)**2))/n)**(1/2)
    return (current*s)+m       
scaled_x=np.copy(x)

scaled_x[:,0]=z_score(x[:,0],m)
scaled_x[:,1]=z_score(x[:,1],m)
scaled_x[:,2]=z_score(x[:,2],m)
scaled_x[:,3]=z_score(x[:,3],m)
scaled_x[:,4]=z_score(x[:,4],m)
scaled_x[:,5]=z_score(x[:,5],m)
print(scaled_x)
def cost(x,y,w,b,m,l):
    loss=sum(-y*(np.log(prediction(x,w,b)+1e-15))-(1-y)*(np.log(1-prediction(x,w,b)+1e-15)))
    total=loss/m
    reg=(l/(m*2)) * np.sum(w**2)
    return total +reg

def grad(w,b,x,y,a,l,m):
    dj_dw=np.copy(w)
    dj_db=sum(prediction(x,w,b)-y)

    dj_dw=((1/m)*np.dot(x.T,(prediction(x,w,b)-y)))+((l/m)*w)
    b=b-a*dj_db/m
    w=w-a*dj_dw
    return w,b

cost_history=[]
t1=time.time()
print(cost(scaled_x,y,w,b,m,0.1))
for i in range (1000000):
    if i%10000==0:
        cost_history.append(cost(scaled_x,y,w,b,m,0.1))
    w,b=grad(w,b,scaled_x,y,0.2,0.1,m)  
t2=time.time()    
print(cost(scaled_x,y,w,b,m,0.1))
print(t2-t1)

# -------------------------------------------------------
# 1. learning curve
# -------------------------------------------------------
def plot_cost_clean(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, color='#2c3e50', linewidth=2.5)
    plt.title("(Learning Curve)", fontsize=14, fontweight='bold')
    plt.xlabel("(Iteration)", fontsize=12)
    plt.ylabel("(Cost)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# -------------------------------------------------------
# 2. Accuracy
# -------------------------------------------------------
 
def plot_accuracy_pie(x, y, w, b):
    f_wb = prediction(x,w,b)
    predictions = (f_wb >= 0.5).astype(int)
    correct = np.sum(predictions == y)
    wrong = len(y) - correct
    accuracy = (correct / len(y)) * 100
    
    
    labels = [f'Correct({correct})', f'Wrong ({wrong})']
    sizes = [correct, wrong]
    colors = ['#27ae60', '#c0392b'] 
    explode = (0.1, 0)  
    
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 14})
    plt.title(f"final module accuracy: {accuracy:.2f}%", fontsize=16, fontweight='bold')
    plt.show()
   
# -------------------------------------------------------
# 3. (Decision Boundary) 
# -------------------------------------------------------
 

def plot_decision_boundaries(x_scaled, y, w, b):
    
    feature_names = ["dough weight"," cheese weight", "cheese price"," peperoni weight", "vegetables price"," pizza price"]
    

    price_col_index = 5 
    w_price = w[price_col_index]
    
 
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel() 
    
    
    for i in range(5): 
        ax = axes[i]
        
        
        ax.scatter(x_scaled[y==0, i], x_scaled[y==0, price_col_index], 
                   c='blue', label='Fair price', alpha=0.7, edgecolors='k')
        ax.scatter(x_scaled[y==1, i], x_scaled[y==1, price_col_index], 
                   c='red', label='Expensive price', alpha=0.7, edgecolors='k')
        
        
        
        x_min, x_max = x_scaled[:, i].min(), x_scaled[:, i].max()
        x_line = np.linspace(x_min, x_max, 100)
        
        
        if abs(w_price) > 1e-5: 
            y_line = (-b - w[i] * x_line) / w_price
            ax.plot(x_line, y_line, c='black', linewidth=3, linestyle='--', label='decision boundary')
        ax.set_title(f"{feature_names[i]} vs {feature_names[price_col_index]}")
        ax.set_xlabel(feature_names[i] + " (Scaled)")
        ax.set_ylabel("Price (Scaled)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# Calling function
# ==========================================


plot_cost_clean(cost_history)


plot_accuracy_pie(scaled_x, y, w, b)

plot_decision_boundaries(scaled_x, y, w, b)