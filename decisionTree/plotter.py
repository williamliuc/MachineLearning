#根据坐标点画图的第一个版本，可实现带箭头的注解绘制功能
import matplotlib.pyplot as plt  

#定义文本框和箭头格式
decisionNode=dict(boxstyle="sawtooth",fc="0.8") 
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

#绘制带箭头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
	createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,\
		textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

def createPlot():
	fig=plt.figure(1,facecolor='white')#figure第一个参数为图像框标题，第二个参数是背景颜色
	fig.clf()
	createPlot.ax1=plt.subplot(111,frameon=False)
	plotNode('decisionNode',(0.5,0.1),(0.1,0.5),decisionNode)#第一个坐标为箭头结束位置，第二个坐标是箭头起始地址
	plotNode('leafNode',(0.8,0.1),(0.3,0.8),leafNode)
	plt.show()

if __name__=="__main__":
	createPlot() 
