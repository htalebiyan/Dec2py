Dec2py: Decentralized Decision-making tools
=======================================================
Interdepndent network restoration decision-making to enhance resilience
---------------------------------------------------------------------------
.. note::
	This webpage contains code documentation and sample jupyter notebooks for an ongoing project. 
	The documentation is constatntly getting updated and revised, and the codes will be added to
	the website soon.

This website offers code documentation and sample jupyter notebooks for a set of analytic tools that
model the restroation processes in synthetic and real-world interdepndent networks subject to 
external shock and disruption. The codes are develped using Python language. Various restoration 
decision-making models are considered here:

*	**Centralized methods**

	These method solve one optimization problem for the whole
	interdepndent network, which leads to optimal restoration plans. The implication 
	of such models setting is that the decison-maker is one entity who has the complete
	information and authority to restore all layers of the interdepndent network. This 
	kind of methods includes Interdepndent Network Desgin Problem (INDP) :cite:`Gonzalez2016` and 
	time-dependnet INDP (td-INDP) :cite:`Gonzalez2016c`.

*	**Decentralized mrthods**

	These methods model a multi-agent decision-making environment
	where each agent has the authority to restore a single layer and has full information
	about her respective layer and minimal or no information about other layers. It is assumed
	that agents communicatie poorly and not in a timely manner. This kind of methods includes 
	Judgment Call (JC) method :cite:`Talebiyan2019c,Talebiyan2019` with and without 
	Auction-based resource allocations :cite:`Talebiyan2020a` and Interdependent Network 
	Restoration Simultaneous Games (INRSG) and Bayesian Games (INRBG) :cite:`Talebiyan2021c`.


Below, you can find different parts of the website (topics), starting with the demonstration of tools for a toy
interdepndent network followed by a notebook containing the full analysis dashboard. The third section presents 
functions and classes under the hood:

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

