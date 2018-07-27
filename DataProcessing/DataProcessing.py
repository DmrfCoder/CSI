#-*-coding:utf-8-*-

def DataProcessing(Csi_Mat_Path):
    AmplitudesAndPhases=DataProcessing.CsiToAmplitudeAndPhase.getAmplitudesAndPhases(Csi_Mat_Path)
    amplitudes=AmplitudesAndPhases[0]
    phases=AmplitudesAndPhases[1]

    amplitudes=DataProcessing.WeightedMovingAverage.WeightMoveAverage(amplitudes,N=len(amplitudes))

    for i in range(0,len(phases)):
        phases[i]=DataProcessing.PhaseSanitization.PhaseSanitization(phases[i].reshape(30,6),30,6)

    



