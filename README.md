# Kaggle_House_Prices_Transformer_Pytorch
A light-weight Transformer model for Kaggle House Prices Regression Competition

# Kaggle House Prices -Advance Regression Techniques

A simple Pytorch deep learning model for predicting the house price. Lightweight Transformer model is tested for accuracy.
The Transformer architecture is utilized to capture pair-wise affinity of all the features.

<img src="https://user-images.githubusercontent.com/90333984/144088221-27ea75be-cd17-42cb-b37c-ad0e4c2d8e3b.png" width="500">

Network Sturcture:
<img src="https://user-images.githubusercontent.com/90333984/144082762-367e81f0-9e76-4a08-9e97-cf1942b2666a.png" width="400">


<!-- # Table of Contents
1. [Project Objective](#objective)
2. [Python Packages](#packages) -->

## Project Objective <a name="p objective"></a>
This is my first project in PyTorch. The aim of the project is to perform a simple multivariate regression using Transformer model. 

## Python Packages
Get packages by using conda or pip.

1. PyTorch=1.8.0
2. numpy=1.19.2
3. matplotlib=3.3.4
4. pandas=1.1.5
5. seaborn=0.11.2

## Kaggle
Once finished, you can upload your prediction.csv to the kaggle website where you can compare your score with other users.

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0
 style='margin-left:21.6pt;border-collapse:collapse;border:none'>
 <tr>
  <td width=131 rowspan=2 style='width:98.3pt;border:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>Model</span></p>
  </td>
  <td width=262 colspan=2 style='width:196.6pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>5-Fold
  Validation</span></p>
  </td>
  <td width=131 rowspan=2 style='width:98.3pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>Test loss
  (rmse)</span></p>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-size:9.0pt;font-family:"Times New Roman",serif'>(on
  official test dataset)</span></p>
  </td>
 </tr>
 <tr>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph style='text-indent:0cm'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>Train loss (rmse)</span></p>
  </td>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph style='text-indent:0cm'><span lang=EN-US
  style='font-family:"Times New Roman",serif'>Test loss (rmse)</span></p>
  </td>
 </tr>
 <tr>
  <td width=131 valign=top style='width:98.3pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>MLP (1
  Block)</span></p>
  </td>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>0.127530</span></p>
  </td>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>0.140763</span></p>
  </td>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>0.15460</span></p>
  </td>
 </tr>
 <tr>
  <td width=131 valign=top style='width:98.3pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>MLP (2
  Blocks)</span></p>
  </td>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>0.108675</span></p>
  </td>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>0.163794</span></p>
  </td>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>0.15125</span></p>
  </td>
 </tr>
 <tr>
  <td width=131 valign=top style='width:98.3pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><span lang=EN-US style='font-family:"Times New Roman",serif'>Ours</span></p>
  </td>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><b><span lang=EN-US style='font-family:"Times New Roman",serif'>0.017307</span></b></p>
  </td>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><b><span lang=EN-US style='font-family:"Times New Roman",serif'>0.129986</span></b></p>
  </td>
  <td width=131 valign=top style='width:98.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoListParagraph align=center style='text-align:center;text-indent:
  0cm'><b><span lang=EN-US style='font-family:"Times New Roman",serif'>0.12760</span></b></p>
  </td>
 </tr>
</table>


## Acknowledgement
You need to have more than 4GB GPU memory to train the model with default settings, or you need to change batchsize or the network sturctures.
