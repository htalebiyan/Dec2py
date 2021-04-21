Dec2py: Decentralized Decision-making tools
=======================================================
Decision support tools for the interdependent network restoration to enhance resilience
-------------------------------------------------------------------------------------------
.. note::
	This webpage contains code documentation and sample jupyter notebooks for an ongoing project. The documentation is constantly getting updated and revised, and the codes will be added to the website soon.

This website offers code documentation and sample jupyter notebooks for a set of analytic tools that model the restoration processes in synthetic and real-world interdependent networks subject to external shock and disruption. The codes are developed using Python language. Various restoration decision-making models are considered here:

*	**Centralized methods**

	These methods solve one optimization problem for the whole interdependent network, which leads to optimal restoration plans. The implication 
	of such models setting is that the decision-maker is one entity who has the complete information and authority to restore all layers of the interdependent network. This kind of methods includes Interdependent Network Design Problem (INDP) :cite:`Gonzalez2016` and time-dependent INDP (td-INDP) :cite:`Gonzalez2016c`.

*	**Decentralized methods**

	These methods model a multi-agent decision-making environment where each agent has the authority to restore a single layer, has complete information about her respective layer, and minimal or no information about other layers. It is assumed that agents communicate poorly and not on time. The decentralized methods include Judgment Call (JC) method  :cite:`Talebiyan2019c,Talebiyan2019` with and without auction-based resource allocations :cite:`Talebiyan2020a` and Interdependent Network Restoration Simultaneous Games (INRSG) and Bayesian Games (INRBG) :cite:`Talebiyan2021c`.


Below, you can find different parts of the website (topics), starting with demonstrating tools for an interdependent toy network followed by a notebook containing the complete analysis dashboard. The third section presents functions and classes under the hood:

.. toctree::
   :maxdepth: 2
   :caption: Topics
	
   run_samples.ipynb
   run_main.ipynb
   backend.rst

   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

To-do works
===================
.. todolist::

Bibliography
===================
.. bibliography:: biblio.bib

