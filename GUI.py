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
import wx
import wx.lib.buttons
import numpy as np
import pickle
import csv
import matplotlib.pyplot as pp
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

import neural_network
import NN_learning



def create(parent):
    return Frame1(parent)

[wxID_FRAME1, wxID_FRAME1BTNCHECK, wxID_FRAME1BTNFILE, wxID_FRAME1BTNLEARN,
 wxID_FRAME1BTNSALOAD, wxID_FRAME1BTNSASAVE, wxID_FRAME1BTNSAVIS,
 wxID_FRAME1BTNSTRUCTANALYSIS, wxID_FRAME1BTNVISUALIZE,
 wxID_FRAME1CHBATCHSIZE, wxID_FRAME1CHCSV, wxID_FRAME1CHDATAREPRESENT,
 wxID_FRAME1GGLPROGESS, wxID_FRAME1GGSAPROGRESS, wxID_FRAME1PANEL1,
 wxID_FRAME1PANEL2, wxID_FRAME1PANEL3, wxID_FRAME1PANELPARAMETERS,
 wxID_FRAME1RBSAVISPARAMS, wxID_FRAME1SLIDERTESTSETSIZE,
 wxID_FRAME1STBATCHSIZE, wxID_FRAME1STCHECK, wxID_FRAME1STDATAREPRESENT,
 wxID_FRAME1STEXAMPLE, wxID_FRAME1STHIDDEN, wxID_FRAME1STHIDDENNUMBER,
 wxID_FRAME1STITERATION, wxID_FRAME1STLEARNING, wxID_FRAME1STLERNINGRATE,
 wxID_FRAME1STMOMENTUM, wxID_FRAME1STNEPOCHS, wxID_FRAME1STNUMBEROFBATCHES,
 wxID_FRAME1STPARAMETERS, wxID_FRAME1STRANDINITNUMBER,
 wxID_FRAME1STREGULARIZATION, wxID_FRAME1STREPRESENTATION,
 wxID_FRAME1STREPRNUMBER, wxID_FRAME1STSAPROGRESS, wxID_FRAME1STSIGSLOPE,
 wxID_FRAME1STSTRUCTUREANALYSIS, wxID_FRAME1STTESTSETPERCENT,
 wxID_FRAME1STTESTSETSIZE, wxID_FRAME1STVISUALIZATION,
 wxID_FRAME1STWEIGHTSLIMIT, wxID_FRAME1TXTEXAMPLE, wxID_FRAME1TXTFILENAME,
 wxID_FRAME1TXTFILEPATH, wxID_FRAME1TXTHIDDEN, wxID_FRAME1TXTHIDDENNUMBER,
 wxID_FRAME1TXTITERATION, wxID_FRAME1TXTLEARNINGRATE, wxID_FRAME1TXTMOMENTUM,
 wxID_FRAME1TXTNEPOCHS, wxID_FRAME1TXTNUMBEROFBATCHES,
 wxID_FRAME1TXTRANDINITNUMBER, wxID_FRAME1TXTREGULARIZATION,
 wxID_FRAME1TXTREPRESENTATION, wxID_FRAME1TXTREPRNUMBER,
 wxID_FRAME1TXTSIGMOIDSLOPE, wxID_FRAME1TXTWEIGHTSLIMIT,
] = [wx.NewId() for _init_ctrls in range(60)]

class Frame1(wx.Frame):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_FRAME1, name='', parent=prnt,
              pos=wx.Point(317, 12), size=wx.Size(617, 485),
              style=wx.DEFAULT_FRAME_STYLE, title='Frame1')
        self.SetClientSize(wx.Size(609, 458))

        self.panelParameters = wx.Panel(id=wxID_FRAME1PANELPARAMETERS,
              name=u'panelParameters', parent=self, pos=wx.Point(8, 8),
              size=wx.Size(256, 440), style=wx.TAB_TRAVERSAL)

        self.panel2 = wx.Panel(id=wxID_FRAME1PANEL2, name='panel2', parent=self,
              pos=wx.Point(272, 8), size=wx.Size(112, 112),
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
              label=u'25%', name='stTestSetPercent',
              parent=self.panelParameters, pos=wx.Point(232, 216), size=wx.Size(24, 16), style=0)

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
              pos=wx.Point(136, 48), size=wx.Size(88, 21), style=0, value=u'6')

        self.txtHidden = wx.TextCtrl(id=wxID_FRAME1TXTHIDDEN, name=u'txtHidden',
              parent=self.panelParameters, pos=wx.Point(136, 72),
              size=wx.Size(88, 21), style=0, value=u'8')

        self.txtWeightsLimit = wx.TextCtrl(id=wxID_FRAME1TXTWEIGHTSLIMIT,
              name=u'txtWeightsLimit', parent=self.panelParameters,
              pos=wx.Point(136, 96), size=wx.Size(32, 21), style=0,
              value=u'0.5')

        self.txtLearningRate = wx.TextCtrl(id=wxID_FRAME1TXTLEARNINGRATE,
              name=u'txtLearningRate', parent=self.panelParameters,
              pos=wx.Point(136, 120), size=wx.Size(32, 21), style=0,
              value=u'1')

        self.txtSigmoidSlope = wx.TextCtrl(id=wxID_FRAME1TXTSIGMOIDSLOPE,
              name=u'txtSigmoidSlope', parent=self.panelParameters,
              pos=wx.Point(136, 144), size=wx.Size(32, 21), style=0,
              value=u'1.5')

        self.txtRegularization = wx.TextCtrl(id=wxID_FRAME1TXTREGULARIZATION,
              name=u'txtRegularization', parent=self.panelParameters,
              pos=wx.Point(136, 168), size=wx.Size(32, 21), style=0,
              value=u'0')

        self.txtMomentum = wx.TextCtrl(id=wxID_FRAME1TXTMOMENTUM,
              name=u'txtMomentum', parent=self.panelParameters,
              pos=wx.Point(136, 192), size=wx.Size(32, 21), style=0,
              value=u'0')

        self.sliderTestSetSize = wx.Slider(id=wxID_FRAME1SLIDERTESTSETSIZE,
              maxValue=100, minValue=0, name=u'sliderTestSetSize',
              parent=self.panelParameters, pos=wx.Point(128, 216),
              size=wx.Size(96, 24), style=wx.SL_HORIZONTAL, value=25)
        self.sliderTestSetSize.SetLabel(u'')
        self.sliderTestSetSize.Bind(wx.EVT_SCROLL, self.OnSliderTestSetSizeScroll)

        self.chDataRepresent = wx.Choice(choices=['complex', 'separate'],
              id=wxID_FRAME1CHDATAREPRESENT, name=u'chDataRepresent',
              parent=self.panelParameters, pos=wx.Point(136, 240),
              size=wx.Size(80, 21), style=0)
        self.chDataRepresent.SetSelection(1)

        self.txtNEpochs = wx.TextCtrl(id=wxID_FRAME1TXTNEPOCHS,
              name=u'txtNEpochs', parent=self.panelParameters, pos=wx.Point(136,
              264), size=wx.Size(64, 21), style=0, value=u'100')

        self.chBatchSize = wx.Choice(choices=['full batch', 'mini batch',
              'online'], id=wxID_FRAME1CHBATCHSIZE, name='chBatchSize',
              parent=self.panelParameters, pos=wx.Point(136, 288),
              size=wx.Size(80, 21), style=0)
        self.chBatchSize.SetSelection(1)
        self.chBatchSize.Bind(wx.EVT_CHOICE, self.OnChBatchSizeChoice,
                              id=wxID_FRAME1CHBATCHSIZE)

        self.txtNumberOfBatches = wx.TextCtrl(id=wxID_FRAME1TXTNUMBEROFBATCHES,
              name=u'txtNumberOfBatches', parent=self.panelParameters,
              pos=wx.Point(136, 312), size=wx.Size(104, 21), style=0,
              value=u'10')
        self.txtNumberOfBatches.SetEditable(True)
        self.txtNumberOfBatches.Enable(False)

        self.stNumberOfBatches = wx.StaticText(id=wxID_FRAME1STNUMBEROFBATCHES,
              label=u'Number of batches', name=u'stNumberOfBatches',
              parent=self.panelParameters, pos=wx.Point(32, 312),
              size=wx.Size(92, 13), style=0)
        self.stNumberOfBatches.Enable(False)

        self.stLearning = wx.StaticText(id=wxID_FRAME1STLEARNING,
              label=u'Learning', name=u'stLearning', parent=self.panel2,
              pos=wx.Point(16, 8), size=wx.Size(75, 21), style=0)
        self.stLearning.SetFont(wx.Font(13, wx.SWISS, wx.NORMAL, wx.BOLD, False,
              u'MS Shell Dlg 2'))

        self.txtFilePath = wx.TextCtrl(id=wxID_FRAME1TXTFILEPATH,
              name=u'txtFilePath', parent=self.panelParameters, pos=wx.Point(32,
              392), size=wx.Size(180, 21), style=0,
              value=u'c:\\SNN\\Learn_data\\ilashevskaya.csv')

        self.btnFile = wx.Button(id=wxID_FRAME1BTNFILE, label=u'File',
              name=u'btnFile', parent=self.panelParameters, pos=wx.Point(32,
              352), size=wx.Size(72, 24), style=0)
        self.btnFile.Bind(wx.EVT_BUTTON, self.OnBtnFileButton,
              id=wxID_FRAME1BTNFILE)

        self.btnLearn = wx.Button(id=wxID_FRAME1BTNLEARN, label=u'Learn',
              name=u'btnLearn', parent=self.panel2, pos=wx.Point(16, 32),
              size=wx.Size(80, 24), style=0)
        self.btnLearn.Bind(wx.EVT_BUTTON, self.OnBtnLearnButton,
              id=wxID_FRAME1BTNLEARN)

        self.panel1 = wx.Panel(id=wxID_FRAME1PANEL1, name='panel1', parent=self,
              pos=wx.Point(272, 128), size=wx.Size(328, 320),
              style=wx.TAB_TRAVERSAL)

        self.btnVisualize = wx.Button(id=wxID_FRAME1BTNVISUALIZE,
              label=u'Visualize', name=u'btnVisualize', parent=self.panel2,
              pos=wx.Point(16, 80), size=wx.Size(80, 24), style=0)
        self.btnVisualize.Bind(wx.EVT_BUTTON, self.OnBtnVisualizeButton,
              id=wxID_FRAME1BTNVISUALIZE)
        self.btnVisualize.fig = False

        self.stStructureAnalysis = wx.StaticText(id=wxID_FRAME1STSTRUCTUREANALYSIS,
              label=u'Structure Analysis', name=u'stStructureAnalysis',
              parent=self.panel1, pos=wx.Point(72, 16), size=wx.Size(185, 24),
              style=0)
        self.stStructureAnalysis.SetFont(wx.Font(16, wx.SWISS, wx.NORMAL,
              wx.BOLD, False, u'Arial'))

        self.btnStructAnalysis = wx.Button(id=wxID_FRAME1BTNSTRUCTANALYSIS,
              label=u'Analyse', name=u'btnStructAnalysis', parent=self.panel1,
              pos=wx.Point(16, 128), size=wx.Size(64, 32), style=0)
        self.btnStructAnalysis.Bind(wx.EVT_BUTTON,
              self.OnBtnStructAnalysisButton, id=wxID_FRAME1BTNSTRUCTANALYSIS)

        self.txtReprNumber = wx.TextCtrl(id=wxID_FRAME1TXTREPRNUMBER,
              name=u'txtReprNumber', parent=self.panel1, pos=wx.Point(280, 56),
              size=wx.Size(40, 21), style=0, value=u'')

        self.txtHiddenNumber = wx.TextCtrl(id=wxID_FRAME1TXTHIDDENNUMBER,
              name=u'txtHiddenNumber', parent=self.panel1, pos=wx.Point(280,
              80), size=wx.Size(40, 21), style=0, value=u'')

        self.txtRandInitNumber = wx.TextCtrl(id=wxID_FRAME1TXTRANDINITNUMBER,
              name=u'txtRandInitNumber', parent=self.panel1, pos=wx.Point(280,
              104), size=wx.Size(40, 21), style=0, value=u'')

        self.stReprNumber = wx.StaticText(id=wxID_FRAME1STREPRNUMBER,
              label=u'Maximum number of neurons in the representaiton layer',
              name=u'stReprNumber', parent=self.panel1, pos=wx.Point(8, 56),
              size=wx.Size(270, 13), style=0)

        self.stHiddenNumber = wx.StaticText(id=wxID_FRAME1STHIDDENNUMBER,
              label=u'Maximum number of neurons  in the hidden layer ',
              name=u'stHiddenNumber', parent=self.panel1, pos=wx.Point(40, 80),
              size=wx.Size(237, 13), style=0)

        self.stRandInitNumber = wx.StaticText(id=wxID_FRAME1STRANDINITNUMBER,
              label=u'Number of random weights initializations',
              name=u'stRandInitNumber', parent=self.panel1, pos=wx.Point(80,
              104), size=wx.Size(193, 13), style=0)

        self.stSAprogress = wx.StaticText(id=wxID_FRAME1STSAPROGRESS,
              label=u'progress', name=u'stSAprogress', parent=self.panel1,
              pos=wx.Point(168, 120), size=wx.Size(50, 16), style=0)
        self.stSAprogress.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.NORMAL,
              False, u'MS Shell Dlg 2'))

        self.ggSAprogress = wx.Gauge(id=wxID_FRAME1GGSAPROGRESS,
              name=u'ggSAprogress', parent=self.panel1, pos=wx.Point(96, 136),
              range=100, size=wx.Size(224, 16), style=wx.GA_HORIZONTAL)

        self.btnSAsave = wx.Button(id=wxID_FRAME1BTNSASAVE, label=u'save',
              name=u'btnSAsave', parent=self.panel1, pos=wx.Point(16, 168),
              size=wx.Size(72, 40), style=0)
        self.btnSAsave.Bind(wx.EVT_BUTTON, self.OnBtnSAsaveButton,
              id=wxID_FRAME1BTNSASAVE)

        self.btnSAload = wx.Button(id=wxID_FRAME1BTNSALOAD, label=u'load',
              name=u'btnSAload', parent=self.panel1, pos=wx.Point(232, 168),
              size=wx.Size(64, 23), style=0)
        self.btnSAload.Bind(wx.EVT_BUTTON, self.OnBtnSAloadButton,
              id=wxID_FRAME1BTNSALOAD)

        self.btnSAVis = wx.Button(id=wxID_FRAME1BTNSAVIS, label=u'Visualize',
              name=u'btnSAVis', parent=self.panel1, pos=wx.Point(240, 264),
              size=wx.Size(75, 40), style=0)
        self.btnSAVis.Bind(wx.EVT_BUTTON, self.OnBtnSAVisButton,
              id=wxID_FRAME1BTNSAVIS)

        self.rbSAVisParams = wx.RadioBox(choices=['train',
              'train (overfitting)', 'test', 'test (overfitting)'],
              id=wxID_FRAME1RBSAVISPARAMS, label=u'parameters',
              majorDimension=2, name=u'rbSAVisParams', parent=self.panel1,
              pos=wx.Point(8, 248), size=wx.Size(224, 64),
              style=wx.RA_SPECIFY_COLS)

        self.stVisualization = wx.StaticText(id=wxID_FRAME1STVISUALIZATION,
              label=u'Visualization', name=u'stVisualization',
              parent=self.panel1, pos=wx.Point(104, 224), size=wx.Size(112, 24),
              style=0)
        self.stVisualization.SetFont(wx.Font(15, wx.SWISS, wx.NORMAL, wx.NORMAL,
              False, u'MS Shell Dlg 2'))

        self.chCSV = wx.CheckBox(id=wxID_FRAME1CHCSV, label=u'csv',
              name=u'chCSV', parent=self.panel1, pos=wx.Point(96, 192),
              size=wx.Size(70, 16), style=0)
        self.chCSV.SetValue(False)

        self.txtFileName = wx.TextCtrl(id=wxID_FRAME1TXTFILENAME,
              name=u'txtFileName', parent=self.panel1, pos=wx.Point(96, 168),
              size=wx.Size(96, 16), style=0, value=u'')

        self.panel3 = wx.Panel(id=wxID_FRAME1PANEL3, name='panel3', parent=self,
              pos=wx.Point(392, 8), size=wx.Size(208, 112),
              style=wx.TAB_TRAVERSAL)

        self.stCheck = wx.StaticText(id=wxID_FRAME1STCHECK,
              label=u'Check result', name=u'stCheck', parent=self.panel3,
              pos=wx.Point(48, 8), size=wx.Size(101, 19), style=0)
        self.stCheck.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, False,
              u'MS Shell Dlg 2'))

        self.btnCheck = wx.Button(id=wxID_FRAME1BTNCHECK, label=u'Check',
              name=u'btnCheck', parent=self.panel3, pos=wx.Point(24, 80),
              size=wx.Size(160, 23), style=0)
        self.btnCheck.Bind(wx.EVT_BUTTON, self.OnBtnCheckButton,
              id=wxID_FRAME1BTNCHECK)

        self.txtExample = wx.TextCtrl(id=wxID_FRAME1TXTEXAMPLE,
              name=u'txtExample', parent=self.panel3, pos=wx.Point(16, 48),
              size=wx.Size(80, 21), style=0, value=u'')

        self.txtIteration = wx.TextCtrl(id=wxID_FRAME1TXTITERATION,
              name=u'txtIteration', parent=self.panel3, pos=wx.Point(112, 48),
              size=wx.Size(80, 21), style=0, value=u'')

        self.stExample = wx.StaticText(id=wxID_FRAME1STEXAMPLE,
              label=u'example', name=u'stExample', parent=self.panel3,
              pos=wx.Point(40, 32), size=wx.Size(41, 13), style=0)

        self.stIteration = wx.StaticText(id=wxID_FRAME1STITERATION,
              label=u'iteration', name=u'stIteration', parent=self.panel3,
              pos=wx.Point(128, 32), size=wx.Size(41, 13), style=0)

        self.ggLprogess = wx.Gauge(id=wxID_FRAME1GGLPROGESS, name=u'ggLprogess',
              parent=self.panel2, pos=wx.Point(8, 64), range=100,
              size=wx.Size(96, 8), style=wx.GA_HORIZONTAL)

    def __init__(self, parent):
            self._init_ctrls(parent)

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
                defaultDir=os.getcwd(),
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
        gauge = self.ggLprogess
        # Set file name
        file = self.txtFilePath.GetValue()

        # Perform learning
        (self.btnLearn.J,
         self.btnLearn.J_test,
         self.btnLearn.theta_history,
         time_ext, time_int) = NN_learning.SNN(hidden_1, hidden_2, epsilon, alpha,
                                               S, R, M, number_of_epochs,
                                               number_of_batches, data_proportion,
                                               online_learning, data_representation,
                                               file, gauge)
        self.txtIteration.SetValue(str(len(self.btnLearn.J) - 1))  # Set default value of  iteration
        event.Skip()


    def OnBtnVisualizeButton(self, event):
        if self.btnVisualize.fig:
            dlg = wx.MessageDialog(None, 'Close the previous figure', 'Warning!',
                                   wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
        else:
            self.btnVisualize.fig = pp.figure(1)
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
            self.btnVisualize.fig = False
        event.Skip()


    def OnBtnStructAnalysisButton(self, event):
        # Take parameters
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
        # Set file name
        file = self.txtFilePath.GetValue()
        # Set SA parameters
        hidden_1_max = int(self.txtReprNumber.GetValue())
        hidden_2_max = int(self.txtHiddenNumber.GetValue())
        num_init = int(self.txtRandInitNumber.GetValue())     # number of random initializations
        self.ggSAprogress.SetRange(hidden_1_max * hidden_2_max)    # det gauge range
        # Prepare date from given file
        [item, rel, attr, batch_size,
         number_of_batches, training_ex_idx,
         test_item_set, test_rel_set, test_attr_set] = NN_learning.Prepare_Learning(epsilon, number_of_epochs,
                                                                                    number_of_batches, data_proportion,
                                                                                    online_learning, data_representation, file)
        # Prepare arrays to fill with error values
        SA_train = np.zeros((hidden_1_max, hidden_2_max))
        SA_train_of = np.zeros((hidden_1_max, hidden_2_max))
        SA_test = np.zeros((hidden_1_max, hidden_2_max))
        SA_test_of = np.zeros((hidden_1_max, hidden_2_max))

        for i in range(hidden_2_max):      # Loop over the hidden layer
            hidden_2 = [i+1]                 # Set number of neurons in the second layer(hidden)

            for j in range(hidden_1_max):  # Loop over  the representaton layer
                hidden_1 = [j+1]             # Set number of neurons in the first layer(representation)
                # Compute errors over several random initializations
                [train_init, train_init_of,
                 test_init, test_init_of] = NN_learning.Rand_Inits(num_init, alpha, R, S, M, hidden_1, hidden_2,
                                                                   epsilon, batch_size, item, rel, attr,
                                                                   data_representation, data_proportion,
                                                                   number_of_epochs, number_of_batches,
                                                                   training_ex_idx, test_item_set,
                                                                   test_rel_set, test_attr_set)
                 # take average error value
                SA_train[j, i] = np.average(train_init)
                SA_train_of[j, i] = np.average(train_init_of)
                SA_test[j, i] = np.average(test_init)
                SA_test_of[j, i] = np.average(test_init_of)
                # Show progress
                progress = ((hidden_2[0]-1)*hidden_1_max) + hidden_1[0]
                self.ggSAprogress.SetValue(progress)
                # save output variables for transmitting
                self.btnStructAnalysis.J_SA = [SA_train, SA_train_of, SA_test, SA_test_of]
                self.btnStructAnalysis.cfg = dict(epsilon=epsilon, alpha=alpha, S=S, R=R, M=M, number_of_epochs=number_of_epochs,
                                                  number_of_batches=number_of_batches, data_proportion=data_proportion,
                                                  online_learning=online_learning, file=file, hidden_1_max=hidden_1_max,
                                                  hidden_2_max=hidden_2_max, num_init=num_init)
        event.Skip()


    def OnBtnSAsaveButton(self, event):
        [SA_train, SA_train_of, SA_test, SA_test_of] = self.btnStructAnalysis.J_SA
        csv_opt = self.chCSV.GetValue()
        file_name = self.txtFileName.GetValue()
        cfg = self.btnStructAnalysis.cfg
        NN_learning.save_SA_results(SA_train, SA_train_of, SA_test, SA_test_of,
                                    cfg, file_name, csv_opt)
        event.Skip()

    def OnBtnSAloadButton(self, event):
        dlg = wx.FileDialog(self, message="Choose a file",
                            defaultDir=os.getcwd(),
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

    def OnBtnSAVisButton(self, event):
        # Choose matrix to visualize
        if self.rbSAVisParams.GetSelection() == 0:
            J_SA = self.btnStructAnalysis.J_SA[0]
        elif self.rbSAVisParams.GetSelection() == 1:
            J_SA = self.btnStructAnalysis.J_SA[1]
        elif self.rbSAVisParams.GetSelection() == 2:
            J_SA = self.btnStructAnalysis.J_SA[2]
        elif self.rbSAVisParams.GetSelection() == 3:
            J_SA = self.btnStructAnalysis.J_SA[3]
        # Visualization
        NN_learning.disp_struct_analysis(J_SA)
        event.Skip()


    def OnBtnCheckButton(self, event):
        example = int(self.txtExample.GetValue())
        epoch = int(self.txtIteration.GetValue())
        file = self.txtFilePath.GetValue()
        hidden_1 = map(int, self.txtRepresentation.GetValue().split(','))
        hidden_2 = map(int, self.txtHidden.GetValue().split(','))
        S = float(self.txtSigmoidSlope.GetValue())
        theta_history = self.btnLearn.theta_history
        check_results = neural_network.check_result(example, epoch, file, hidden_1,
                                                    hidden_2, theta_history, S)
        caption = 'Check results'
        dlg = wx.MessageDialog(None, check_results, caption, wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()
        event.Skip()


if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = create(None)
    frame.Show()

    app.MainLoop()
