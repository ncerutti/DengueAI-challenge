import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score 

def evaluate(y,yhat,figname='figure',visualize=True,):
    #calculate scores
    R2, RMSE, MAE = calc_score(y,yhat) 
    print (f" R2 = {R2} \n RMSE:{RMSE} \n MAE:{MAE}")
    
    #make figures
    fig, ax = plt.subplots(2,1, figsize=(12,6))
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.4)
    #Scatter plot y vs yhat
    ax[0].plot(y,yhat,'.')
    ax[0].plot([0,y.max()],[0,y.max()],'-')
    ax[0].set(xlabel='True Total Cases', ylabel='Predicted Total Cases')
    ax[0].text(0.1,0.75, 'RMSE:%5.2f\nR2:%5.2f\nMAE:%5.2f'%(RMSE,R2,MAE), transform=ax[0].transAxes)
    
    #TS plot
    ax[1].plot(range(0,len(y)),y,':b',label='true')
    ax[1].plot(range(0,len(y)),yhat,'-r',label='predicted')
    ax[1].set(xlabel='Week',ylabel='Total Cases')
    plt.legend()

    fig.savefig(figname + '.png')
    
    return {'R2':R2,'RMSE':RMSE,'MAE':MAE}

def calc_score(y,yhat):
    R2 = r2_score(y, yhat)
    RMSE = mean_squared_error(y,yhat,squared=False)
    MAE = mean_absolute_error(y,yhat)
    return (R2,RMSE,MAE)