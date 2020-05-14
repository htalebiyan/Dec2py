Input Parameters
=======================================================
.. data:: FILTER_SCE

   (str, optional) The address to the list of scenarios that should be included in the analyses.
   
.. data:: BASE_DIR

   (str) The address to the basic (topology, parameters, etc.) information of the network.  
   
.. data:: DAMAGE_DIR

   (str) The address to damge scenario data.

.. data:: OUTPUT_DIR

   (str) The address to where output are stored.

.. data:: FAIL_SCE_PARAM

   (dict) Informatiom on the ype of the failure scenario (Andres or Wu)
   and network dataset (shelby or synthetic).
   Help:
   For Andres scenario: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
   magnitudes: FAIL_SCE_PARAM['MAGS'].
   For Wu scenario: set range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
   sce range: FAIL_SCE_PARAM['MAGS'].
   For Synthetic nets: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
   configurations: FAIL_SCE_PARAM['MAGS'].
   For random disruption: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
   FAIL_SCE_PARAM['MAGS']=0.
   
.. toctree::
   :maxdepth: 2
   :caption: Topics: