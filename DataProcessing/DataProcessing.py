#-*-coding:utf-8-*-

def DataProcessing(Csi_Mat_Path):

    AmplitudesAndPhases=DataProcessing.CsiToAmplitudeAndPhase.getAmplitudesAndPhases(Csi_Mat_Path)

    amplitudes=AmplitudesAndPhases[0]#n*180
    phases=AmplitudesAndPhases[1]#n*180

    amplitudes=DataProcessing.WeightedMovingAverage.WeightMoveAverage(amplitudes,N=len(amplitudes))

    for i in range(0,len(phases)):
        phases[i]=DataProcessing.PhaseSanitization.PhaseSanitization(phases[i].reshape(30,6),30,6)


    return (amplitudes,phases)
    #amplitudes和amplitudes的维度都是n*180，现在将其写为n*360即可，但是要注意打上label






