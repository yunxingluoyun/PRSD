# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:36:59 2021

@author: xiaohuihui
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)


def save_classification_report(y_true,y_pred,target_names,save_path):
    '''
    # 将分类报告保存至csv文件

    Parameters
    ----------
    y_true : TYPE
        标签.
    y_pred : TYPE
        预测结果.
    target_names : TYPE
        类别名.
    save_path : TYPE
        保存路径.

    Returns
    -------
    TYPE
       分类精度报告.
    
    示例
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> save_path = r'精度评价.csv'
    >>> acc_report_df = save_classification_report(y_true,y_pred,target_names,save_path)
    >>> print(acc_report_df)


    '''
    acc_report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=target_names,output_dict=True)).T
    acc_report_df.iloc[-3,:2]= np.nan
    acc_report_df.iloc[-3,3]= acc_report_df.iloc[-2,3]
    # acc_report_df.iloc[-3,2]= np.nan
    acc_report_df.round(4).to_csv(save_path)
    return acc_report_df.round(4)
