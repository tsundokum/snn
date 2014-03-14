#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Name:        FrameMineSNN.py
# Purpose:
#
# Author:      Ilya Pershin
#
# Created:     2013/08/14
# RCS-ID:      $Id: FrameMineSNN.py $
# Copyright:   (c) 2006
# Licence:     <your licence>
#-----------------------------------------------------------------------------
#Boa:Frame:Frame1
import os
import sys
import wx
import wx.lib.buttons
import numpy as np
import pickle
import csv
import thread
import matplotlib.pyplot as pp
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

import neural_network
import NN_learning
import NN_analysis


def create(parent):
    return Frame1(parent)

[wxID_FRAME1, wxID_FRAME1BTNCHECK, wxID_FRAME1BTNFILE, wxID_FRAME1BTNGMA,
 wxID_FRAME1BTNLEARN, wxID_FRAME1BTNSAANALYSE, wxID_FRAME1BTNSAVIZUALISE,
 wxID_FRAME1BTNSTRUCTANALYSIS, wxID_FRAME1BTNVISUALIZE,
 wxID_FRAME1CHBATCHSIZE, wxID_FRAME1CHCOSTFUNCTION,
 wxID_FRAME1CHDATAREPRESENT, wxID_FRAME1CHEXACTERROREVAL,
 wxID_FRAME1CHTRAINEVAL, wxID_FRAME1GGLPROGESS, wxID_FRAME1PANEL1,
 wxID_FRAME1PANEL2, wxID_FRAME1PANEL3, wxID_FRAME1PANELPARAMETERS,
 wxID_FRAME1SLIDERTESTSETSIZE, wxID_FRAME1STBATCHSIZE, wxID_FRAME1STCHECK,
 wxID_FRAME1STCOSTFUNCTION, wxID_FRAME1STDATAREPRESENT,
 wxID_FRAME1STEXACTERROREVAL, wxID_FRAME1STEXAMPLE, wxID_FRAME1STHIDDEN,
 wxID_FRAME1STHIDDENNUMBER, wxID_FRAME1STITERATION, wxID_FRAME1STLEARNING,
 wxID_FRAME1STLERNINGRATE, wxID_FRAME1STMOMENTUM, wxID_FRAME1STNEPOCHS,
 wxID_FRAME1STNUMBEROFBATCHES, wxID_FRAME1STOUTDIR, wxID_FRAME1STPARAMETERS,
 wxID_FRAME1STPATHFILEDIR, wxID_FRAME1STRANDINITNUMBER,
 wxID_FRAME1STREGULARIZATION, wxID_FRAME1STREMAININGTIME,
 wxID_FRAME1STREPRESENTATION, wxID_FRAME1STREPRNUMBER, wxID_FRAME1STSIGSLOPE,
 wxID_FRAME1STSTRUCTUREANALYSIS, wxID_FRAME1STTESTSETPERCENT,
 wxID_FRAME1STTESTSETSIZE, wxID_FRAME1STTRAINEVAL, wxID_FRAME1STWEIGHTSLIMIT,
 wxID_FRAME1TXTCURFILE, wxID_FRAME1TXTEXAMPLE, wxID_FRAME1TXTFILEPATH,
 wxID_FRAME1TXTHIDDEN, wxID_FRAME1TXTHIDDENRANGE, wxID_FRAME1TXTITERATION,
 wxID_FRAME1TXTLEARNINGRATE, wxID_FRAME1TXTMOMENTUM, wxID_FRAME1TXTNEPOCHS,
 wxID_FRAME1TXTNUMBEROFBATCHES, wxID_FRAME1TXTOUTDIR,
 wxID_FRAME1TXTPATHFILEDIR, wxID_FRAME1TXTRANDINITNUMBER,
 wxID_FRAME1TXTREGULARIZATION, wxID_FRAME1TXTREMAININGTIME,
 wxID_FRAME1TXTREPRESENTATION, wxID_FRAME1TXTREPRRANGE,
 wxID_FRAME1TXTSIGMOIDSLOPE, wxID_FRAME1TXTWEIGHTSLIMIT,
] = [wx.NewId() for _init_ctrls in range(67)]

class Frame1(wx.Frame):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_FRAME1, name='', parent=prnt,
              pos=wx.Point(18, 76), size=wx.Size(672, 476),
              style=wx.DEFAULT_FRAME_STYLE, title='SNN')
        self.SetClientSize(wx.Size(664, 449))

        self.panelParameters = wx.Panel(id=wxID_FRAME1PANELPARAMETERS,
              name=u'panelParameters', parent=self, pos=wx.Point(8, 8),
              size=wx.Size(288, 432), style=wx.TAB_TRAVERSAL)

        self.panel2 = wx.Panel(id=wxID_FRAME1PANEL2, name='panel2', parent=self,
              pos=wx.Point(304, 8), size=wx.Size(112, 120),
              style=wx.TAB_TRAVERSAL)

        self.stParameters = wx.StaticText(id=wxID_FRAME1STPARAMETERS,
              label=u'Parameters', name=u'stParameters',
              parent=self.panelParameters, pos=wx.Point(88, 8), size=wx.Size(97,
              21), style=0)
        self.stParameters.SetFont(wx.Font(13, wx.SWISS, wx.NORMAL, wx.BOLD,
              False, u'MS Shell Dlg 2'))

        self.stRepresentation = wx.StaticText(id=wxID_FRAME1STREPRESENTATION,
              label=u'Representarion layer', name=u'stRepresentation',
              parent=self.panelParameters, pos=wx.Point(24, 48),
              size=wx.Size(102, 13), style=0)

        self.stHidden = wx.StaticText(id=wxID_FRAME1STHIDDEN,
              label=u'Hidden layer', name=u'stHidden',
              parent=self.panelParameters, pos=wx.Point(64, 72),
              size=wx.Size(64, 13), style=0)

        self.stWeightsLimit = wx.StaticText(id=wxID_FRAME1STWEIGHTSLIMIT,
              label=u'Initial weights limitation', name=u'stWeightsLimit',
              parent=self.panelParameters, pos=wx.Point(8, 96),
              size=wx.Size(112, 13), style=0)

        self.stLerningRate = wx.StaticText(id=wxID_FRAME1STLERNINGRATE,
              label=u'Learning rate', name=u'stLerningRate',
              parent=self.panelParameters, pos=wx.Point(56, 120),
              size=wx.Size(65, 13), style=0)

        self.stSigSlope = wx.StaticText(id=wxID_FRAME1STSIGSLOPE,
              label=u'Sigmoid slope', name=u'stSigSlope',
              parent=self.panelParameters, pos=wx.Point(56, 144),
              size=wx.Size(65, 13), style=0)

        self.stRegularization = wx.StaticText(id=wxID_FRAME1STREGULARIZATION,
              label=u'Regularization', name=u'stRegularization',
              parent=self.panelParameters, pos=wx.Point(48, 168),
              size=wx.Size(69, 13), style=0)

        self.stMomentum = wx.StaticText(id=wxID_FRAME1STMOMENTUM,
              label=u'Momentum', name=u'stMomentum',
              parent=self.panelParameters, pos=wx.Point(64, 192),
              size=wx.Size(53, 13), style=0)

        self.stTestSetSize = wx.StaticText(id=wxID_FRAME1STTESTSETSIZE,
              label=u'Test set size', name=u'stTestSetSize',
              parent=self.panelParameters, pos=wx.Point(56, 216),
              size=wx.Size(61, 13), style=0)

        self.stTestSetPercent = wx.StaticText(id=wxID_FRAME1STTESTSETPERCENT,
              label=u'', name='stTestSetPercent', parent=self.panelParameters,
              pos=wx.Point(232, 216), size=wx.Size(24, 16), style=0)

        self.stDataRepresent = wx.StaticText(id=wxID_FRAME1STDATAREPRESENT,
              label=u'Data representation', name=u'stDataRepresent',
              parent=self.panelParameters, pos=wx.Point(24, 240),
              size=wx.Size(98, 13), style=0)

        self.stNEpochs = wx.StaticText(id=wxID_FRAME1STNEPOCHS,
              label=u'Number of epochs', name=u'stNEpochs',
              parent=self.panelParameters, pos=wx.Point(32, 264),
              size=wx.Size(88, 13), style=0)

        self.stBatchSize = wx.StaticText(id=wxID_FRAME1STBATCHSIZE,
              label=u'Batch type', name=u'stBatchSize',
              parent=self.panelParameters, pos=wx.Point(64, 288),
              size=wx.Size(57, 13), style=0)

        self.txtRepresentation = wx.TextCtrl(id=wxID_FRAME1TXTREPRESENTATION,
              name=u'txtRepresentation', parent=self.panelParameters,
              pos=wx.Point(136, 48), size=wx.Size(88, 21), style=0, value=u'')

        self.txtHidden = wx.TextCtrl(id=wxID_FRAME1TXTHIDDEN, name=u'txtHidden',
              parent=self.panelParameters, pos=wx.Point(136, 72),
              size=wx.Size(88, 21), style=0, value=u'')

        self.txtWeightsLimit = wx.TextCtrl(id=wxID_FRAME1TXTWEIGHTSLIMIT,
              name=u'txtWeightsLimit', parent=self.panelParameters,
              pos=wx.Point(136, 96), size=wx.Size(32, 21), style=0, value=u'')

        self.txtLearningRate = wx.TextCtrl(id=wxID_FRAME1TXTLEARNINGRATE,
              name=u'txtLearningRate', parent=self.panelParameters,
              pos=wx.Point(136, 120), size=wx.Size(32, 21), style=0, value=u'')

        self.txtSigmoidSlope = wx.TextCtrl(id=wxID_FRAME1TXTSIGMOIDSLOPE,
              name=u'txtSigmoidSlope', parent=self.panelParameters,
              pos=wx.Point(136, 144), size=wx.Size(32, 21), style=0, value=u'')

        self.txtRegularization = wx.TextCtrl(id=wxID_FRAME1TXTREGULARIZATION,
              name=u'txtRegularization', parent=self.panelParameters,
              pos=wx.Point(136, 168), size=wx.Size(32, 21), style=0, value=u'')

        self.txtMomentum = wx.TextCtrl(id=wxID_FRAME1TXTMOMENTUM,
              name=u'txtMomentum', parent=self.panelParameters,
              pos=wx.Point(136, 192), size=wx.Size(32, 21), style=0, value=u'')

        self.sliderTestSetSize = wx.Slider(id=wxID_FRAME1SLIDERTESTSETSIZE,
              maxValue=100, minValue=0, name=u'sliderTestSetSize',
              parent=self.panelParameters, pos=wx.Point(128, 216),
              size=wx.Size(96, 24), style=wx.SL_HORIZONTAL, value=0)
        self.sliderTestSetSize.Bind(wx.EVT_SCROLL,
              self.OnSliderTestSetSizeScroll)

        self.chDataRepresent = wx.Choice(choices=['complex', 'separate'],
              id=wxID_FRAME1CHDATAREPRESENT, name=u'chDataRepresent',
              parent=self.panelParameters, pos=wx.Point(136, 240),
              size=wx.Size(80, 21), style=0)

        self.txtNEpochs = wx.TextCtrl(id=wxID_FRAME1TXTNEPOCHS,
              name=u'txtNEpochs', parent=self.panelParameters, pos=wx.Point(136,
              264), size=wx.Size(64, 21), style=0, value=u'')

        self.chBatchSize = wx.Choice(choices=['full batch', 'mini batch',
              'online'], id=wxID_FRAME1CHBATCHSIZE, name='chBatchSize',
              parent=self.panelParameters, pos=wx.Point(136, 288),
              size=wx.Size(80, 21), style=0)
        self.chBatchSize.Bind(wx.EVT_CHOICE, self.OnChBatchSizeChoice,
              id=wxID_FRAME1CHBATCHSIZE)

        self.txtNumberOfBatches = wx.TextCtrl(id=wxID_FRAME1TXTNUMBEROFBATCHES,
              name=u'txtNumberOfBatches', parent=self.panelParameters,
              pos=wx.Point(136, 312), size=wx.Size(40, 21), style=0, value=u'')

        self.stNumberOfBatches = wx.StaticText(id=wxID_FRAME1STNUMBEROFBATCHES,
              label=u'Number of batches', name=u'stNumberOfBatches',
              parent=self.panelParameters, pos=wx.Point(32, 312),
              size=wx.Size(92, 13), style=0)

        self.stLearning = wx.StaticText(id=wxID_FRAME1STLEARNING,
              label=u'Learning', name=u'stLearning', parent=self.panel2,
              pos=wx.Point(16, 8), size=wx.Size(75, 21), style=0)
        self.stLearning.SetFont(wx.Font(13, wx.SWISS, wx.NORMAL, wx.BOLD, False,
              u'MS Shell Dlg 2'))

        self.txtFilePath = wx.TextCtrl(id=wxID_FRAME1TXTFILEPATH,
              name=u'txtFilePath', parent=self.panelParameters, pos=wx.Point(16,
              392), size=wx.Size(180, 21), style=0, value=u'')

        self.btnFile = wx.Button(id=wxID_FRAME1BTNFILE, label=u'File',
              name=u'btnFile', parent=self.panelParameters, pos=wx.Point(208,
              392), size=wx.Size(72, 24), style=0)
        self.btnFile.Bind(wx.EVT_BUTTON, self.OnBtnFileButton,
              id=wxID_FRAME1BTNFILE)

        self.btnLearn = wx.Button(id=wxID_FRAME1BTNLEARN, label=u'Learn',
              name=u'btnLearn', parent=self.panel2, pos=wx.Point(16, 32),
              size=wx.Size(80, 24), style=0)
        self.btnLearn.Bind(wx.EVT_BUTTON, self.OnBtnLearnButton,
              id=wxID_FRAME1BTNLEARN)

        self.panel1 = wx.Panel(id=wxID_FRAME1PANEL1, name='panel1', parent=self,
              pos=wx.Point(424, 8), size=wx.Size(232, 432),
              style=wx.TAB_TRAVERSAL)

        self.btnVisualize = wx.Button(id=wxID_FRAME1BTNVISUALIZE,
              label=u'Visualize', name=u'btnVisualize', parent=self.panel2,
              pos=wx.Point(16, 80), size=wx.Size(80, 24), style=0)
        self.btnVisualize.Bind(wx.EVT_BUTTON, self.OnBtnVisualizeButton,
              id=wxID_FRAME1BTNVISUALIZE)

        self.stStructureAnalysis = wx.StaticText(id=wxID_FRAME1STSTRUCTUREANALYSIS,
              label=u'Structure Analysis', name=u'stStructureAnalysis',
              parent=self.panel1, pos=wx.Point(16, 8), size=wx.Size(185, 24),
              style=0)
        self.stStructureAnalysis.SetFont(wx.Font(16, wx.SWISS, wx.NORMAL,
              wx.BOLD, False, u'Arial'))

        self.btnStructAnalysis = wx.Button(id=wxID_FRAME1BTNSTRUCTANALYSIS,
              label=u'SA', name=u'btnStructAnalysis', parent=self.panel1,
              pos=wx.Point(8, 280), size=wx.Size(96, 24), style=0)
        self.btnStructAnalysis.Bind(wx.EVT_BUTTON,
              self.OnBtnStructAnalysisButton, id=wxID_FRAME1BTNSTRUCTANALYSIS)

        self.txtReprRange = wx.TextCtrl(id=wxID_FRAME1TXTREPRRANGE,
              name=u'txtReprRange', parent=self.panel1, pos=wx.Point(176, 48),
              size=wx.Size(40, 24), style=0, value=u'')

        self.txtHiddenRange = wx.TextCtrl(id=wxID_FRAME1TXTHIDDENRANGE,
              name=u'txtHiddenRange', parent=self.panel1, pos=wx.Point(176, 88),
              size=wx.Size(40, 24), style=0, value=u'')

        self.txtRandInitNumber = wx.TextCtrl(id=wxID_FRAME1TXTRANDINITNUMBER,
              name=u'txtRandInitNumber', parent=self.panel1, pos=wx.Point(176,
              128), size=wx.Size(40, 24), style=0, value=u'')

        self.stReprNumber = wx.StaticText(id=wxID_FRAME1STREPRNUMBER,
              label=u'Range of numbers of neurons\n in the representaiton layer',
              name=u'stReprNumber', parent=self.panel1, pos=wx.Point(16, 48),
              size=wx.Size(144, 26), style=0)

        self.stHiddenNumber = wx.StaticText(id=wxID_FRAME1STHIDDENNUMBER,
              label=u'Range of numbers of neurons  \nin the hidden layer ',
              name=u'stHiddenNumber', parent=self.panel1, pos=wx.Point(16, 88),
              size=wx.Size(150, 26), style=0)

        self.stRandInitNumber = wx.StaticText(id=wxID_FRAME1STRANDINITNUMBER,
              label=u'Number of random \nweights initializations',
              name=u'stRandInitNumber', parent=self.panel1, pos=wx.Point(56,
              128), size=wx.Size(101, 26), style=0)

        self.panel3 = wx.Panel(id=wxID_FRAME1PANEL3, name='panel3', parent=self,
              pos=wx.Point(304, 136), size=wx.Size(112, 304),
              style=wx.TAB_TRAVERSAL)

        self.stCheck = wx.StaticText(id=wxID_FRAME1STCHECK,
              label=u'Check result', name=u'stCheck', parent=self.panel3,
              pos=wx.Point(8, 8), size=wx.Size(101, 19), style=0)
        self.stCheck.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, False,
              u'MS Shell Dlg 2'))

        self.btnCheck = wx.Button(id=wxID_FRAME1BTNCHECK, label=u'Check',
              name=u'btnCheck', parent=self.panel3, pos=wx.Point(8, 136),
              size=wx.Size(96, 23), style=0)
        self.btnCheck.Bind(wx.EVT_BUTTON, self.OnBtnCheckButton,
              id=wxID_FRAME1BTNCHECK)

        self.txtExample = wx.TextCtrl(id=wxID_FRAME1TXTEXAMPLE,
              name=u'txtExample', parent=self.panel3, pos=wx.Point(16, 48),
              size=wx.Size(80, 21), style=0, value=u'0')

        self.txtIteration = wx.TextCtrl(id=wxID_FRAME1TXTITERATION,
              name=u'txtIteration', parent=self.panel3, pos=wx.Point(16, 96),
              size=wx.Size(80, 21), style=0, value=u'')

        self.stExample = wx.StaticText(id=wxID_FRAME1STEXAMPLE,
              label=u'example', name=u'stExample', parent=self.panel3,
              pos=wx.Point(40, 32), size=wx.Size(41, 13), style=0)

        self.stIteration = wx.StaticText(id=wxID_FRAME1STITERATION,
              label=u'iteration', name=u'stIteration', parent=self.panel3,
              pos=wx.Point(40, 80), size=wx.Size(41, 13), style=0)

        self.ggLprogess = wx.Gauge(id=wxID_FRAME1GGLPROGESS, name=u'ggLprogess',
              parent=self.panel2, pos=wx.Point(8, 64), range=100,
              size=wx.Size(96, 8), style=wx.GA_HORIZONTAL)

        self.stCostFunction = wx.StaticText(id=wxID_FRAME1STCOSTFUNCTION,
              label=u'Cost Function', name=u'stCostFunction',
              parent=self.panelParameters, pos=wx.Point(56, 344),
              size=wx.Size(67, 13), style=0)

        self.chCostFunction = wx.Choice(choices=['least squares',
              'cross-entropy'], id=wxID_FRAME1CHCOSTFUNCTION,
              name=u'chCostFunction', parent=self.panelParameters,
              pos=wx.Point(136, 336), size=wx.Size(96, 21), style=0)
        self.chCostFunction.SetSelection(0)

        self.stExactErrorEval = wx.StaticText(id=wxID_FRAME1STEXACTERROREVAL,
              label=u'Exact error evaluation', name=u'stExactErrorEval',
              parent=self.panelParameters, pos=wx.Point(16, 368),
              size=wx.Size(108, 13), style=0)

        self.chExactErrorEval = wx.CheckBox(id=wxID_FRAME1CHEXACTERROREVAL,
              label=u'', name=u'chExactErrorEval', parent=self.panelParameters,
              pos=wx.Point(144, 368), size=wx.Size(16, 13), style=0)

        self.stTrainEval = wx.StaticText(id=wxID_FRAME1STTRAINEVAL,
              label=u'Additional train-only evaluation', name=u'stTrainEval',
              parent=self.panel1, pos=wx.Point(16, 168), size=wx.Size(150, 13),
              style=0)

        self.chTrainEval = wx.CheckBox(id=wxID_FRAME1CHTRAINEVAL, label=u'',
              name=u'chTrainEval', parent=self.panel1, pos=wx.Point(176, 168),
              size=wx.Size(16, 13), style=0)
        self.chTrainEval.SetValue(True)

        self.txtPathFileDir = wx.TextCtrl(id=wxID_FRAME1TXTPATHFILEDIR,
              name=u'txtPathFileDir', parent=self.panel1, pos=wx.Point(8, 208),
              size=wx.Size(212, 21), style=0, value=u'')

        self.stPathFileDir = wx.StaticText(id=wxID_FRAME1STPATHFILEDIR,
              label=u'path to file or directory', name=u'stPathFileDir',
              parent=self.panel1, pos=wx.Point(64, 192), size=wx.Size(120, 13),
              style=0)

        self.txtOutDir = wx.TextCtrl(id=wxID_FRAME1TXTOUTDIR, name=u'txtOutDir',
              parent=self.panel1, pos=wx.Point(8, 248), size=wx.Size(216, 21),
              style=0, value=u'')

        self.stOutDir = wx.StaticText(id=wxID_FRAME1STOUTDIR,
              label=u'output directory', name=u'stOutDir', parent=self.panel1,
              pos=wx.Point(72, 232), size=wx.Size(79, 13), style=0)

        self.txtRemainingTime = wx.TextCtrl(id=wxID_FRAME1TXTREMAININGTIME,
              name=u'txtRemainingTime', parent=self.panel1, pos=wx.Point(8,
              352), size=wx.Size(216, 19), style=0, value=u'')

        self.stRemainingTime = wx.StaticText(id=wxID_FRAME1STREMAININGTIME,
              label=u'progress', name=u'stRemainingTime', parent=self.panel1,
              pos=wx.Point(88, 312), size=wx.Size(48, 13), style=0)

        self.btnSAVizualise = wx.Button(id=wxID_FRAME1BTNSAVIZUALISE,
              label=u'Vizualise', name=u'btnSAVizualise', parent=self.panel1,
              pos=wx.Point(128, 384), size=wx.Size(88, 32), style=0)
        self.btnSAVizualise.Bind(wx.EVT_BUTTON, self.OnBtnSAVizualiseButton,
              id=wxID_FRAME1BTNSAVIZUALISE)

        self.btnSAAnalyse = wx.Button(id=wxID_FRAME1BTNSAANALYSE,
              label=u'SA table', name=u'btnSAAnalyse', parent=self.panel1,
              pos=wx.Point(16, 384), size=wx.Size(96, 32), style=0)
        self.btnSAAnalyse.Bind(wx.EVT_BUTTON, self.OnButton1Button,
              id=wxID_FRAME1BTNSAANALYSE)

        self.btnGMA = wx.Button(id=wxID_FRAME1BTNGMA, label=u'GMA',
              name=u'btnGMA', parent=self.panel1, pos=wx.Point(120, 280),
              size=wx.Size(91, 23), style=0)
        self.btnGMA.Bind(wx.EVT_BUTTON, self.OnBtnGMAButton,
              id=wxID_FRAME1BTNGMA)

        self.txtCurFile = wx.TextCtrl(id=wxID_FRAME1TXTCURFILE,
              name=u'txtCurFile', parent=self.panel1, pos=wx.Point(8, 328),
              size=wx.Size(216, 19), style=0, value=u'')

    def __init__(self, parent):
        self._init_ctrls(parent)
        # load last parameters
        if 'last_cfg.pkl' in os.listdir(sys.path[0]):
            with open(sys.path[0]+'\last_cfg.pkl', 'rb') as f:
                cfg = pickle.load(f)
        else:
        # set default parameters
            cfg = dict(hidden_1=[6], hidden_2=[8], epsilon=0.5, alpha=0.3,
                   S=3, R=0, M=0, number_of_epochs=50, number_of_batches=8,
                   data_proportion=0.25, online_learning='on',
                   data_representation='complex', cost_function='mean_squares',
                   exact_error_eval=True, file_name=sys.path[0]+'\\01.csv',
                   hidden_1_range=[3,10], hidden_2_range=[3,15], num_init=5,
                   f=u'', out_dir=u'', train_eval=True)
        # set values in vidgets
        self.txtRepresentation.SetValue(unicode(cfg['hidden_1'])[1:-1])
        self.txtHidden.SetValue(unicode(cfg['hidden_2'])[1:-1])
        self.txtWeightsLimit.SetValue(unicode(cfg['epsilon']))
        self.txtLearningRate.SetValue(unicode(cfg['alpha']))
        self.txtSigmoidSlope.SetValue(unicode(cfg['S']))
        self.txtRegularization.SetValue(unicode(cfg['R']))
        self.txtMomentum.SetValue(unicode(cfg['M']))
        if cfg['data_representation'] != 'large':
            self.chDataRepresent.SetSelection(['complex','separate'].index(cfg['data_representation']))
        self.stTestSetPercent.SetLabel(unicode(int(cfg['data_proportion']*100))+' %')
        self.sliderTestSetSize.SetValue(int(cfg['data_proportion']*100))
        self.txtNEpochs.SetValue(unicode(cfg['number_of_epochs']))
        if cfg['online_learning'] == 'on':
            self.chBatchSize.SetSelection(2)
        else:
            if cfg['number_of_batches'] == 1:
                self.chBatchSize.SetSelection(0)
            else:
                self.chBatchSize.SetSelection(1)
        self.txtNumberOfBatches.SetValue(unicode(cfg['number_of_batches']))
        self.txtFilePath.SetValue(unicode(cfg['file_name']))
        self.chCostFunction.SetSelection(['mean_squares','cross_entropy'].index(cfg['cost_function']))
        self.chExactErrorEval.SetValue(cfg['exact_error_eval'])
        self.txtReprRange.SetValue(unicode(cfg['hidden_1_range'])[1:-1])
        self.txtHiddenRange.SetValue(unicode(cfg['hidden_2_range'])[1:-1])
        self.txtRandInitNumber.SetValue(unicode(cfg['num_init']))
        self.chTrainEval.SetValue(cfg['train_eval'])
        self.txtPathFileDir.SetValue(unicode(cfg['f']))
        self.txtOutDir.SetValue(unicode(cfg['out_dir']))


    def OnSliderTestSetSizeScroll(self, event):
        self.stTestSetPercent.SetLabel(str(self.sliderTestSetSize.GetValue())+'%')
        event.Skip()

    def OnChBatchSizeChoice(self, event):
        if self.chBatchSize.GetSelection() == 1:
            self.txtNumberOfBatches.Enable(True)
            self.stNumberOfBatches.Enable(True)
        else:
            self.txtNumberOfBatches.Enable(False)
            self.stNumberOfBatches.Enable(False)
        event.Skip()

    def OnBtnFileButton(self, event):
        dlg = wx.FileDialog(
                self, message="Choose a file",
                defaultDir=sys.path[0],
                defaultFile="",
                style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
                )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.txtFilePath.SetValue(path)
        dlg.Destroy()
        event.Skip()

    def OnBtnLearnButton(self, event):
        # Take parameters:
        # Set net structure
        hidden_1 = map(int, self.txtRepresentation.GetValue().split(','))
        hidden_2 = map(int, self.txtHidden.GetValue().split(','))
        # Set learning parameters
        epsilon = float(self.txtWeightsLimit.GetValue())
        alpha = float(self.txtLearningRate.GetValue())
        S = float(self.txtSigmoidSlope.GetValue())
        R = float(self.txtRegularization.GetValue())
        M = float(self.txtMomentum.GetValue())
        # Set data preperation parameters
        data_proportion = float(self.sliderTestSetSize.GetValue()) / 100
        if self.chDataRepresent.GetSelection() == 0:
            data_representation = 'complex'
        elif self.chDataRepresent.GetSelection() == 1:
            data_representation = 'separate'
        number_of_epochs = int(self.txtNEpochs.GetValue())
        if self.chBatchSize.GetSelection() == 0:
            number_of_batches = 1
            online_learning = ''
            self.txtNumberOfBatches.SetValue('1')
        elif self.chBatchSize.GetSelection() == 1:
            number_of_batches = int(self.txtNumberOfBatches.GetValue())
            online_learning = ''
        elif self.chBatchSize.GetSelection() == 2:
            online_learning = 'on'
            self.txtNumberOfBatches.SetValue('number of examples')
            number_of_batches = 'none'
        if self.chCostFunction.GetSelection() == 0:
            cost_function = 'mean_squares'
        elif self.chCostFunction.GetSelection() == 1:
            cost_function = 'cross_entropy'
        exact_error_eval = self.chExactErrorEval.GetValue()
        gauge = self.ggLprogess
        file_name = self.txtFilePath.GetValue() # Set file name
        # SA parameters for saving
        hidden_1_range = map(int, self.txtReprRange.GetValue().split(','))
        hidden_2_range = map(int, self.txtHiddenRange.GetValue().split(','))
        num_init = int(self.txtRandInitNumber.GetValue())     # number of random initializations
        f = self.txtPathFileDir.GetValue()
        out_dir = self.txtOutDir.GetValue()
        train_eval = self.chTrainEval.GetValue()

        # Save current cfg
        cfg = dict(hidden_1=hidden_1, hidden_2=hidden_2, epsilon=epsilon,
                   alpha=alpha, S=S, R=R, M=M, number_of_epochs=number_of_epochs,
                   number_of_batches=number_of_batches, data_proportion=data_proportion,
                   online_learning=online_learning, data_representation=data_representation,
                   cost_function=cost_function,exact_error_eval=exact_error_eval,
                   file_name=file_name, hidden_1_range=hidden_1_range,
                   hidden_2_range=hidden_2_range, num_init=num_init, f=f,
                   out_dir=out_dir, train_eval=train_eval)
        NN_learning.save_cfg(cfg, sys.path[0]+'\last', txt=False)

        # Perform learning
        (self.btnLearn.J,
         self.btnLearn.J_test,
         self.btnLearn.theta_history,
         time_ext, time_int) = NN_learning.SNN(hidden_1, hidden_2, epsilon, alpha,
                                               S, R, M, number_of_epochs,
                                               number_of_batches, data_proportion,
                                               online_learning, data_representation,
                                               cost_function, exact_error_eval,
                                               file_name, gauge)
        self.txtIteration.SetValue(str(len(self.btnLearn.J) - 1))  # Set default value of  iteration
        event.Skip()


    def OnBtnVisualizeButton(self, event):
##        if self.btnVisualize.fig:
##            dlg = wx.MessageDialog(None, 'Close the previous figure', 'Warning!',
##                                   wx.OK | wx.ICON_INFORMATION)
##            dlg.ShowModal()
##            dlg.Destroy()
##        else:
        J = self.btnLearn.J
        J_test = self.btnLearn.J_test
        num_iter = range(len(J))
        pp.subplot(211)
        pp.plot(num_iter, J)
        pp.ylabel('Error')
        pp.title('Training error')
        pp.subplot(212)
        pp.plot(num_iter, J_test)
        pp.xlabel('Iteration')
        pp.ylabel('Error')
        pp.title('Test error')
        pp.show()
        event.Skip()


    def OnBtnStructAnalysisButton(self, event):
        # Take parameters:
        # Set net structure for saving
        hidden_1 = map(int, self.txtRepresentation.GetValue().split(','))
        hidden_2 = map(int, self.txtHidden.GetValue().split(','))
        # Set learning parameters
        epsilon = float(self.txtWeightsLimit.GetValue())
        alpha = float(self.txtLearningRate.GetValue())
        S = float(self.txtSigmoidSlope.GetValue())
        R = float(self.txtRegularization.GetValue())
        M = float(self.txtMomentum.GetValue())
        # Set data preperation parameters
        data_proportion = float(self.sliderTestSetSize.GetValue()) / 100
        if self.chDataRepresent.GetSelection() == 0:
            data_representation = 'complex'
        elif self.chDataRepresent.GetSelection() == 1:
            data_representation = 'separate'
        number_of_epochs = int(self.txtNEpochs.GetValue())
        if self.chBatchSize.GetSelection() == 0:
            number_of_batches = 1
            online_learning = ''
            self.txtNumberOfBatches.SetValue('1')
        elif self.chBatchSize.GetSelection() == 1:
            number_of_batches = int(self.txtNumberOfBatches.GetValue())
            online_learning = ''
        elif self.chBatchSize.GetSelection() == 2:
            online_learning = 'on'
            self.txtNumberOfBatches.SetValue('number of examples')
            number_of_batches = 'none'
        if self.chCostFunction.GetSelection() == 0:
            cost_function = 'mean_squares'
        elif self.chCostFunction.GetSelection() == 1:
            cost_function = 'cross_entropy'
        exact_error_eval = self.chExactErrorEval.GetValue()
        file_name = self.txtFilePath.GetValue() # Set file name for saving
        # Set SA parameters
        hidden_1_range = map(int, self.txtReprRange.GetValue().split(','))
        hidden_2_range = map(int, self.txtHiddenRange.GetValue().split(','))
        num_init = int(self.txtRandInitNumber.GetValue())     # number of random initializations
        f = self.txtPathFileDir.GetValue()
        out_dir = self.txtOutDir.GetValue()
        train_eval = self.chTrainEval.GetValue()
        # Save current cfg
        cfg = dict(hidden_1=hidden_1, hidden_2=hidden_2, epsilon=epsilon,
                   alpha=alpha, S=S, R=R, M=M, number_of_epochs=number_of_epochs,
                   number_of_batches=number_of_batches, data_proportion=data_proportion,
                   online_learning=online_learning, data_representation=data_representation,
                   cost_function=cost_function,exact_error_eval=exact_error_eval,
                   file_name=file_name, hidden_1_range=hidden_1_range,
                   hidden_2_range=hidden_2_range, num_init=num_init, f=f,
                   out_dir=out_dir, train_eval=train_eval)
        NN_learning.save_cfg(cfg, sys.path[0]+'\last', txt=False)

        thread.start_new_thread(NN_analysis.full_SA,
                                (hidden_1_range, hidden_2_range, num_init, epsilon,
                                 alpha, S, R, M, number_of_epochs, number_of_batches,
                                 data_proportion, online_learning, data_representation,
                                 cost_function, exact_error_eval, f, out_dir, train_eval,
                                 self))
        event.Skip()


    def OnBtnSAloadButton(self, event):
        dlg = wx.FileDialog(self, message="Choose a file",
                            defaultDir=sys.path[0],
                            defaultFile="",
                            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            # Load SA-results
            SA_file = path
            loaded_SA = open(SA_file, 'r')
            J_SA = pickle.load(loaded_SA)
            loaded_SA.close()
            self.btnStructAnalysis.J_SA = J_SA
        dlg.Destroy()
        event.Skip()


    def OnBtnCheckButton(self, event):
        example = int(self.txtExample.GetValue())
        epoch = int(self.txtIteration.GetValue())
        file_name = self.txtFilePath.GetValue()
        hidden_1 = map(int, self.txtRepresentation.GetValue().split(','))
        hidden_2 = map(int, self.txtHidden.GetValue().split(','))
        S = float(self.txtSigmoidSlope.GetValue())
        theta_history = self.btnLearn.theta_history
        check_results = neural_network.check_result(example, epoch, file_name, hidden_1,
                                                    hidden_2, theta_history, S)
        caption = 'Check results'
        dlg = wx.MessageDialog(None, check_results, caption, wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()
        event.Skip()


    def OnBtnSAVizualiseButton(self, event):
        # Get file through dialog
        dlg = wx.FileDialog(self, message="Choose a file",
                            defaultDir=sys.path[0],
                            defaultFile="",
                            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            loaded_SA = open(path, 'r')
            J_SA = pickle.load(loaded_SA)
            loaded_SA.close()
        dlg.Destroy()
        # Visualization
        NN_learning.disp_struct_analysis(J_SA)
        event.Skip()


    def OnButton1Button(self, event):
        table_dir = self.txtOutDir.GetValue()
        NN_analysis.fill_table_SA(str(table_dir))
        event.Skip()


    def OnBtnGMAButton(self, event):
        # Take parameters:
        # Set net structure for saving
        hidden_1 = map(int, self.txtRepresentation.GetValue().split(','))
        hidden_2 = map(int, self.txtHidden.GetValue().split(','))
        # Set learning parameters
        epsilon = float(self.txtWeightsLimit.GetValue())
        alpha = float(self.txtLearningRate.GetValue())
        S = float(self.txtSigmoidSlope.GetValue())
        R = float(self.txtRegularization.GetValue())
        M = float(self.txtMomentum.GetValue())
        # Set data preperation parameters
        data_proportion = float(self.sliderTestSetSize.GetValue()) / 100
        data_representation = 'large'
        number_of_epochs = int(self.txtNEpochs.GetValue())
        if self.chBatchSize.GetSelection() == 0:
            number_of_batches = 1
            online_learning = ''
            self.txtNumberOfBatches.SetValue('1')
        elif self.chBatchSize.GetSelection() == 1:
            number_of_batches = int(self.txtNumberOfBatches.GetValue())
            online_learning = ''
        elif self.chBatchSize.GetSelection() == 2:
            online_learning = 'on'
            self.txtNumberOfBatches.SetValue('number of examples')
            number_of_batches = 'none'
        if self.chCostFunction.GetSelection() == 0:
            cost_function = 'mean_squares'
        elif self.chCostFunction.GetSelection() == 1:
            cost_function = 'cross_entropy'
        exact_error_eval = self.chExactErrorEval.GetValue()
        file_name = self.txtFilePath.GetValue() # Set file name for saving
        # Set SA parameters
        hidden_1_range = map(int, self.txtReprRange.GetValue().split(','))
        hidden_2_range = map(int, self.txtHiddenRange.GetValue().split(','))
        num_init = int(self.txtRandInitNumber.GetValue())     # number of random initializations
        f = self.txtPathFileDir.GetValue()
        out_dir = self.txtOutDir.GetValue()
        train_eval = self.chTrainEval.GetValue()
        # Save current cfg
        cfg = dict(hidden_1=hidden_1, hidden_2=hidden_2, epsilon=epsilon,
                   alpha=alpha, S=S, R=R, M=M, number_of_epochs=number_of_epochs,
                   number_of_batches=number_of_batches, data_proportion=data_proportion,
                   online_learning=online_learning, data_representation=data_representation,
                   cost_function=cost_function,exact_error_eval=exact_error_eval,
                   file_name=file_name, hidden_1_range=hidden_1_range,
                   hidden_2_range=hidden_2_range, num_init=num_init, f=f,
                   out_dir=out_dir, train_eval=train_eval)
        NN_learning.save_cfg(cfg, sys.path[0]+'\last', txt=False)
        # Perform GMA
        theta_list = NN_analysis.GMA(epsilon, alpha, S, R, M, number_of_epochs,
                                                 number_of_batches, data_proportion,
                                                 online_learning, data_representation,
                                                 cost_function, exact_error_eval,
                                                 hidden_1_range, hidden_2_range, num_init,
                                                 f, out_dir, self)
        # fill the table
        NN_analysis.fill_table_GMA(theta_list, file_name, S, hidden_1, hidden_2,
                                   out_dir, str(f))
        event.Skip()


if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = create(None)
    frame.Show()
    app.MainLoop()
