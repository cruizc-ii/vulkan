
        set Ec  24099792.53022731
        set Ic  -0.00702222198998826
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 719.2258654122501
        set lambda 16.258656181392723
        set alpha 0.13
        set Icrit 0.0002164522288459893
        set stable True
        source imk_sections.tcl
        set secTagCol 100001
        set secTagBeam 200001
        set imkMat 300001
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  1  0 12 $secTagCol
        element elasticBeamColumn 2 12 15 1e6 1.0 -0.00702222198998826 2

        set Ec  24099792.53022731
        set Ic  -0.00702222198998826
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 719.2258654122501
        set lambda 16.258656181392723
        set alpha 0.13
        set Icrit 0.0002164522288459893
        set stable True
        source imk_sections.tcl
        set secTagCol 100003
        set secTagBeam 200003
        set imkMat 300003
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  3  15 3 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.00702222198998826
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 719.2258654122501
        set lambda 16.258656181392723
        set alpha 0.13
        set Icrit 0.0002164522288459893
        set stable True
        source imk_sections.tcl
        set secTagCol 100004
        set secTagBeam 200004
        set imkMat 300004
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  4  1 13 $secTagCol
        element elasticBeamColumn 5 13 18 1e6 1.0 -0.00702222198998826 2

        set Ec  24099792.53022731
        set Ic  -0.00702222198998826
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 719.2258654122501
        set lambda 16.258656181392723
        set alpha 0.13
        set Icrit 0.0002164522288459893
        set stable True
        source imk_sections.tcl
        set secTagCol 100006
        set secTagBeam 200006
        set imkMat 300006
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  6  18 4 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.005166950341854762
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 539.4193990591875
        set lambda 16.258518396422254
        set alpha 0.13
        set Icrit 0.00016233917163449197
        set stable True
        source imk_sections.tcl
        set secTagCol 100007
        set secTagBeam 200007
        set imkMat 300007
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  7  3 16 $secTagCol
        element elasticBeamColumn 8 16 21 1e6 1.0 -0.005166950341854762 1

        set Ec  24099792.53022731
        set Ic  -0.005166950341854762
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 539.4193990591875
        set lambda 16.258518396422254
        set alpha 0.13
        set Icrit 0.00016233917163449197
        set stable True
        source imk_sections.tcl
        set secTagCol 100009
        set secTagBeam 200009
        set imkMat 300009
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  9  21 4 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.00702222198998826
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 719.2258654122501
        set lambda 16.258656181392723
        set alpha 0.13
        set Icrit 0.0002164522288459893
        set stable True
        source imk_sections.tcl
        set secTagCol 100010
        set secTagBeam 200010
        set imkMat 300010
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  10  2 14 $secTagCol
        element elasticBeamColumn 11 14 22 1e6 1.0 -0.00702222198998826 2

        set Ec  24099792.53022731
        set Ic  -0.00702222198998826
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 719.2258654122501
        set lambda 16.258656181392723
        set alpha 0.13
        set Icrit 0.0002164522288459893
        set stable True
        source imk_sections.tcl
        set secTagCol 100012
        set secTagBeam 200012
        set imkMat 300012
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  12  22 5 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.005166950341854762
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 539.4193990591875
        set lambda 16.258518396422254
        set alpha 0.13
        set Icrit 0.00016233917163449197
        set stable True
        source imk_sections.tcl
        set secTagCol 100013
        set secTagBeam 200013
        set imkMat 300013
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  13  4 19 $secTagCol
        element elasticBeamColumn 14 19 24 1e6 1.0 -0.005166950341854762 1

        set Ec  24099792.53022731
        set Ic  -0.005166950341854762
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 539.4193990591875
        set lambda 16.258518396422254
        set alpha 0.13
        set Icrit 0.00016233917163449197
        set stable True
        source imk_sections.tcl
        set secTagCol 100015
        set secTagBeam 200015
        set imkMat 300015
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  15  24 5 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.0010377830285780228
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 183.36741050058487
        set lambda 16.258246441350924
        set alpha 0.13
        set Icrit 5.518472931701246e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100016
        set secTagBeam 200016
        set imkMat 300016
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  16  3 17 $secTagCol
        element elasticBeamColumn 17 17 25 1e6 1.0 -0.0010377830285780228 2

        set Ec  24099792.53022731
        set Ic  -0.0010377830285780228
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 183.36741050058487
        set lambda 16.258246441350924
        set alpha 0.13
        set Icrit 5.518472931701246e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100018
        set secTagBeam 200018
        set imkMat 300018
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  18  25 6 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.0010377830285780228
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 183.36741050058487
        set lambda 16.258246441350924
        set alpha 0.13
        set Icrit 5.518472931701246e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100019
        set secTagBeam 200019
        set imkMat 300019
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  19  4 20 $secTagCol
        element elasticBeamColumn 20 20 28 1e6 1.0 -0.0010377830285780228 2

        set Ec  24099792.53022731
        set Ic  -0.0010377830285780228
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 183.36741050058487
        set lambda 16.258246441350924
        set alpha 0.13
        set Icrit 5.518472931701246e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100021
        set secTagBeam 200021
        set imkMat 300021
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  21  28 7 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.0013075007258187674
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 137.52555787543866
        set lambda 16.258211498328485
        set alpha 0.13
        set Icrit 4.138854698775936e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100022
        set secTagBeam 200022
        set imkMat 300022
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  22  6 26 $secTagCol
        element elasticBeamColumn 23 26 31 1e6 1.0 -0.0013075007258187674 1

        set Ec  24099792.53022731
        set Ic  -0.0013075007258187674
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 137.52555787543866
        set lambda 16.258211498328485
        set alpha 0.13
        set Icrit 4.138854698775936e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100024
        set secTagBeam 200024
        set imkMat 300024
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  24  31 7 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.0010377830285780228
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 183.36741050058487
        set lambda 16.258246441350924
        set alpha 0.13
        set Icrit 5.518472931701246e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100025
        set secTagBeam 200025
        set imkMat 300025
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  25  5 23 $secTagCol
        element elasticBeamColumn 26 23 32 1e6 1.0 -0.0010377830285780228 2

        set Ec  24099792.53022731
        set Ic  -0.0010377830285780228
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 183.36741050058487
        set lambda 16.258246441350924
        set alpha 0.13
        set Icrit 5.518472931701246e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100027
        set secTagBeam 200027
        set imkMat 300027
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  27  32 8 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.0013075007258187674
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 137.52555787543866
        set lambda 16.258211498328485
        set alpha 0.13
        set Icrit 4.138854698775936e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100028
        set secTagBeam 200028
        set imkMat 300028
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  28  7 29 $secTagCol
        element elasticBeamColumn 29 29 34 1e6 1.0 -0.0013075007258187674 1

        set Ec  24099792.53022731
        set Ic  -0.0013075007258187674
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 137.52555787543866
        set lambda 16.258211498328485
        set alpha 0.13
        set Icrit 4.138854698775936e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100030
        set secTagBeam 200030
        set imkMat 300030
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  30  34 8 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.00033695246463557297
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 57.59297725157898
        set lambda 16.25815061477997
        set alpha 0.13
        set Icrit 1.733270296784338e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100031
        set secTagBeam 200031
        set imkMat 300031
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  31  6 27 $secTagCol
        element elasticBeamColumn 32 27 35 1e6 1.0 -0.00033695246463557297 2

        set Ec  24099792.53022731
        set Ic  -0.00033695246463557297
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 57.59297725157898
        set lambda 16.25815061477997
        set alpha 0.13
        set Icrit 1.733270296784338e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100033
        set secTagBeam 200033
        set imkMat 300033
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  33  35 9 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.00033695246463557297
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 57.59297725157898
        set lambda 16.25815061477997
        set alpha 0.13
        set Icrit 1.733270296784338e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100034
        set secTagBeam 200034
        set imkMat 300034
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  34  7 30 $secTagCol
        element elasticBeamColumn 35 30 37 1e6 1.0 -0.00033695246463557297 2

        set Ec  24099792.53022731
        set Ic  -0.00033695246463557297
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 57.59297725157898
        set lambda 16.25815061477997
        set alpha 0.13
        set Icrit 1.733270296784338e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100036
        set secTagBeam 200036
        set imkMat 300036
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  36  37 10 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.0005512878133981094
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 57.59297725157898
        set lambda 16.25815061477997
        set alpha 0.13
        set Icrit 1.733270296784338e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100037
        set secTagBeam 200037
        set imkMat 300037
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  37  9 36 $secTagCol
        element elasticBeamColumn 38 36 39 1e6 1.0 -0.0005512878133981094 1

        set Ec  24099792.53022731
        set Ic  -0.0005512878133981094
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 57.59297725157898
        set lambda 16.25815061477997
        set alpha 0.13
        set Icrit 1.733270296784338e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100039
        set secTagBeam 200039
        set imkMat 300039
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  39  39 10 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.00033695246463557297
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 57.59297725157898
        set lambda 16.25815061477997
        set alpha 0.13
        set Icrit 1.733270296784338e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100040
        set secTagBeam 200040
        set imkMat 300040
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  40  8 33 $secTagCol
        element elasticBeamColumn 41 33 40 1e6 1.0 -0.00033695246463557297 2

        set Ec  24099792.53022731
        set Ic  -0.00033695246463557297
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 57.59297725157898
        set lambda 16.25815061477997
        set alpha 0.13
        set Icrit 1.733270296784338e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100042
        set secTagBeam 200042
        set imkMat 300042
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  42  40 11 $secTagCol
        
        set Ec  24099792.53022731
        set Ic  -0.0005512878133981094
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 57.59297725157898
        set lambda 16.25815061477997
        set alpha 0.13
        set Icrit 1.733270296784338e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100043
        set secTagBeam 200043
        set imkMat 300043
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  43  10 38 $secTagCol
        element elasticBeamColumn 44 38 41 1e6 1.0 -0.0005512878133981094 1

        set Ec  24099792.53022731
        set Ic  -0.0005512878133981094
        set theta_y 0.0035412273487457737
        set theta_p 0.017394545260856348
        set theta_pc 0.017311147846284493
        set My 57.59297725157898
        set lambda 16.25815061477997
        set alpha 0.13
        set Icrit 1.733270296784338e-05
        set stable True
        source imk_sections.tcl
        set secTagCol 100045
        set secTagBeam 200045
        set imkMat 300045
        uniaxialMaterial ModIMKPeakOriented $imkMat $Ks $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
        section Aggregator $secTagCol $elasticMatTag P $imkMat Mz
        section Aggregator $secTagBeam $elasticMatTag P $elasticMatTag Mz
        element zeroLengthSection  45  41 11 $secTagCol
        