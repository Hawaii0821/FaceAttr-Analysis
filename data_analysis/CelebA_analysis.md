# 人脸属性数据集CelebA

## CelebA 解释

### 数据集基本描述

* 10177 个名人
* 202599张人脸图片
* 40个属性标记
* 人脸bbox标注框
* 5个人脸特征点坐标

由香港中文大学开放提供，网址在[这里](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)



网盘文件夹描述如下：

![img](https://pic1.zhimg.com/80/v2-e5546e983a24e0e846f11de439068c64_hd.jpg)

* Anno是bbox、landmark及attribute注释文件；**（标注文件夹）**
* Eval是training、validation及testing数据集的划分注释；**（划分描述文件夹）**
* Img则是存放相应的人脸图像，**（人脸图片文件夹）**
* README.txt是CelebA介绍文件；



虽然img文件夹提供了多种格式的图片，我们一般只需要用到 .jpg（img_align_celeba.zip)，文件内部的图片都是有排序的。



### 标签解释



* **人脸标注框**

![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/img/20190309145245.png)



 

* **人脸5个特征点**， align代表着图片经过处理对齐。

![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/img/20190309145516.png)





* 40个特征属性标记

  ![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/img/20190309150157.png)

  可以看出我们的任务是多标签的二分类任务。

  其中每个属性代表的含义是：

1. 5_o_Clock_Shadow：刚长出的双颊胡须
2. Arched_Eyebrows：柳叶眉
3. Attractive：吸引人的
4. Bags_Under_Eyes：眼袋
5. Bald：秃头  ---
6. Bangs：刘海 ---
7. Big_Lips：大嘴唇
8. Big_Nose：大鼻子
9. Black_Hair：黑发
10. Blond_Hair：金发 ---
11. Blurry：模糊的 ---
12. Brown_Hair：棕发
13. Bushy_Eyebrows：浓眉
14. Chubby：圆胖的 --
15. Double_Chin：双下巴 ----
16. Eyeglasses：眼镜 ----
17. Goatee：山羊胡子 -----
18. Gray_Hair：灰发或白发 ----
19. Heavy_Makeup：浓妆
20. High_Cheekbones：高颧骨
21. Male：男性---
22. Mouth_Slightly_Open：微微张开嘴巴
23. Mustache：胡子，髭 ----
24. Narrow_Eyes：细长的眼睛
25. No_Beard：无胡子
26. Oval_Face：椭圆形的脸
27. Pale_Skin：苍白的皮肤 ---
28. Pointy_Nose：尖鼻子
29. Receding_Hairline：发际线后移 -----
30. Rosy_Cheeks：红润的双颊
31. Sideburns：连鬓胡子----
32. Smiling：微笑
33. Straight_Hair：直发
34. Wavy_Hair：卷发
35. Wearing_Earrings：戴着耳环
36. Wearing_Hat：戴着帽子-----
37. Wearing_Lipstick：涂了唇膏
38. Wearing_Necklace：戴着项链
39. Wearing_Necktie：戴着领带
40. Young：年轻人

属性从数值上分类，有名词界定性质和有顺序性质之分，Nominal or ordinal；
从空间位置上来看，有全局和局部属性之分 holistic or local，
CelebA数据集属性就有着全局和局部属性，如何捕获全局高层属性以及定位局部属性的位置并加以分析，是整个任务的核心，在特征提取部分，所有属性能够共享相同的特征。

同时我们发现这40个属性之间，部分属性对存在正或反相关关系，某些属性同属于某一大类；这提示我们在提取特征之后，
我们可以对分类层做对应的有差别的设计，比如将同类属性划分为同一个fc layer进行分类预测。