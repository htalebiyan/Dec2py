import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(context='notebook',style='darkgrid')
#plt.rc('text', usetex=True)
#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
def plot_performance_curves_shelby(df,x='t',y='cost',cost_type='Total',
                            decision_names=['tdindp_results'],
                            auction_type=None,valuation_type=None,
                            ci=None,normalize=False):
    no_resources = df.no_resources.unique().tolist()
    if not auction_type:
        auction_type = df.auction_type.unique().tolist()
    auction_type.remove('')
    if not valuation_type:
        valuation_type = df.valuation_type.unique().tolist()
    valuation_type.remove('')
    T = len(df[x].unique().tolist())
    
    fig, axs = plt.subplots(len(valuation_type), len(no_resources), sharex=True, sharey=True, tight_layout=False)
    for idxnr,nr in enumerate(no_resources):
        for idxvt,vt in enumerate(valuation_type):
            if len(valuation_type)==1 and len(no_resources)==1:
                ax=axs
            elif len(valuation_type)==1:
                ax = axs[idxnr]
            elif len(no_resources)==1:
                ax = axs[idxvt]
            else:
                ax = axs[idxvt,idxnr]
                
            with sns.xkcd_palette(['black',"windows blue",'red',"green"]): #sns.color_palette("muted"):
                ax = sns.lineplot(x=x, y=y, hue="auction_type", style='decision_type',
                    markers=False, ci=ci, ax=ax,legend='full',
                    data=df[(df['cost_type']==cost_type)&
                            (df['decision_type'].isin(decision_names))&
                            (df['no_resources']==nr)&
                            ((df['valuation_type']==vt)|(df['valuation_type']==''))]) 
                ax.set(xlabel=r'time step $t$', ylabel=cost_type+' Cost')
                ax.get_legend().set_visible(False)
                ax.xaxis.set_ticks(np.arange(0,T+1,1.0))   #ax.get_xlim()                          
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5)
    
    if len(valuation_type)==1 and len(no_resources)==1:
        axx=[axs]
        axy=[axs]
    elif len(valuation_type)==1:
        axy = [axs[0]]
        axx = axs
    elif len(no_resources)==1:
        axy = axs
        axx = [axs[0]]
    else:
        axx = axs[0,:]
        axy = axs[:,0]  
    for idx, ax in enumerate(axx):
        ax.set_title(r'Total resources=%d'%(no_resources[idx]))
    for idx, ax in enumerate(axy):
        ax.annotate('Valuation = '+valuation_type[idx], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label, textcoords='offset points', ha='right', va='center', rotation=90) 
        
    plt.savefig('Performance_curves.pdf',dpi=600)  

def plot_performance_curves_synthetic(df,x='t',y='cost',cost_type='Total',ci=None):
    no_resources = df.no_resources.unique().tolist()
    auction_type = df.auction_type.unique().tolist()
    auction_type.remove('')
    valuation_type = df.valuation_type.unique().tolist()
    valuation_type.remove('')
    T = len(df[x].unique().tolist())
    
    hor_grid=[0]
    ver_grid=valuation_type
    fig, axs = plt.subplots(len(ver_grid),len(hor_grid),sharex=True,sharey=True,tight_layout=False)
    for idxnr,nr in enumerate(hor_grid):
        for idxvt,vt in enumerate(ver_grid):
            if len(ver_grid)==1 and len(hor_grid)==1:
                ax=axs
            elif len(ver_grid)==1:
                ax = axs[idxnr]
            elif len(hor_grid)==1:
                ax = axs[idxvt]
            else:
                ax = axs[idxvt,idxnr]

            selected_data=df[(df['cost_type']==cost_type)&
                    ((df['valuation_type']==vt)|(df['valuation_type']==''))]              
            with sns.xkcd_palette(['black',"windows blue",'red',"green"]): #sns.color_palette("muted"):
                ax = sns.lineplot(x=x, y=y, hue="auction_type", style='decision_type',
                    markers=False, ci=ci, ax=ax,legend='full',data=selected_data)
                ax.set(xlabel=r'time step $t$', ylabel=cost_type+' Cost')
                if cost_type=='Under Supply Perc':
                    ax.set(xlabel=r'time step $t$', ylabel='Unmet Demand Ratio')
                ax.get_legend().set_visible(False)
                ax.xaxis.set_ticks(np.arange(0,T+1,1.0))   #ax.get_xlim()                          
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5)
    
    if len(ver_grid)==1 and len(hor_grid)==1:
        axx=[axs]
        axy=[axs]
    elif len(hor_grid)==1:
        axy = [axs[0]]
        axx = axs
    elif len(no_resources)==1:
        axy = axs
        axx = [axs[0]]
    else:
        axx = axs[0,:]
        axy = axs[:,0]  
    for idx, ax in enumerate(axx):
        ax.set_title(r'%d'%(hor_grid[idx]))
    for idx, ax in enumerate(axy):
        ax.annotate('Valuation = '+ver_grid[idx], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label, textcoords='offset points', ha='right', va='center', rotation=90) 
        
    plt.savefig('Performance_curves.pdf',dpi=600)  
    
def plot_relative_performance_shelby(lambda_df,cost_type='Total',lambda_type='U'):    
    no_resources = lambda_df.no_resources.unique().tolist()
    auction_type = lambda_df.auction_type.unique().tolist()
    auction_type.remove('')
    valuation_type = lambda_df.valuation_type.unique().tolist()
    valuation_type.remove('')
    
    fig, axs = plt.subplots(len(valuation_type), len(auction_type),sharex=True, sharey='row',tight_layout=False)
    for idxnr,nr in enumerate(auction_type):   
        for idxvt,vt in enumerate(valuation_type): 
            if len(valuation_type)==1 and len(auction_type)==1:
                ax=axs
            elif len(valuation_type)==1:
                ax = axs[idxnr]
            elif len(auction_type)==1:
                ax = axs[idxvt]
            else:
                ax = axs[idxvt,idxnr]
                                           
            with sns.color_palette("RdYlGn", 8):  #sns.color_palette("YlOrRd", 7)
                ax=sns.barplot(x='no_resources',y='lambda_'+lambda_type,hue="decision_type",
                            data=lambda_df[(lambda_df['cost_type']==cost_type)&
                                                (lambda_df['lambda_'+lambda_type]!='nan')&
                                                ((lambda_df['auction_type']==nr)|(lambda_df['auction_type']==''))&
                                                ((lambda_df['valuation_type']==vt)|(lambda_df['valuation_type']==''))], 
                                linewidth=0.5,edgecolor=[.25,.25,.25],
                                capsize=.05,errcolor=[.25,.25,.25],errwidth=1,ax=ax) 
                ax.get_legend().set_visible(False)
                ax.grid(which='major', axis='y', color=[.75,.75,.75], linewidth=.75)
                ax.set_xlabel(r'\# resources')
                if idxvt!=len(valuation_type)-1:
                    ax.set_xlabel('')
                ax.set_ylabel(r'E[$\lambda_{%s}$], Valuation: %s'%(lambda_type,valuation_type[idxvt]))
                if idxnr!=0:
                    ax.set_ylabel('')
                ax.xaxis.set_label_position('bottom')  
#                ax.xaxis.tick_top()
                ax.set_facecolor('w')
                
    handles, labels = ax.get_legend_handles_labels()   
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels,loc='best',frameon =True,framealpha=0.5,
               ncol=1,bbox_to_anchor=(0.9,0.6)) 
     
    if len(auction_type)==1 and len(valuation_type)==1:
        axx=[axs]
        axy=[axs]
    elif len(auction_type)==1:
        axx = [axs[0]]
        axy = axs
    elif len(valuation_type)==1:
        axx = axs
        axy = [axs[0]]
    else:
        axx = axs[0,:]
        axy = axs[:,0]  
    for idx, ax in enumerate(axx):
        ax.set_title(r'Auction: %s'%(auction_type[idx]))
#    for idx, ax in enumerate(axy):
#        ax.annotate('Valuation:'+valuation_type[idx],xy=(0.1, 0.5),xytext=(-ax.yaxis.labelpad - 5, 0),
#            xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center',rotation=90) 

    plt.savefig('Relative_perforamnce.pdf',dpi=600)

def plot_relative_performance_synthetic(lambda_df,cost_type='Total',lambda_type='U'):     
#    auction_type = lambda_df.auction_type.unique().tolist()
#    auction_type.remove('')
    valuation_type = lambda_df.valuation_type.unique().tolist()
    valuation_type.remove('')
    
    fig, axs = plt.subplots(len(valuation_type),1,sharex=True, sharey='row',tight_layout=False)  
    for idxvt,vt in enumerate(valuation_type): 
        selected_data=lambda_df[(lambda_df['cost_type']==cost_type)&(lambda_df['lambda_'+lambda_type]!='nan')&
                    ((lambda_df['valuation_type']==vt)|(lambda_df['valuation_type']==''))]                                      
        with sns.color_palette("RdYlGn", 8):  #sns.color_palette("YlOrRd", 7)
            ax=sns.barplot(x='auction_type',y='lambda_'+lambda_type,hue="decision_type",
                        data=selected_data,linewidth=0.5,edgecolor=[.25,.25,.25],
                            capsize=.05,errcolor=[.25,.25,.25],errwidth=1,ax=axs) 
            ax.get_legend().set_visible(False)
            ax.grid(which='major', axis='y', color=[.75,.75,.75], linewidth=.75)
            ax.set_xlabel(r'Auction Type')
            if idxvt!=len(valuation_type)-1:
                ax.set_xlabel('')
            ax.set_ylabel(r'E[$\lambda_{%s}$], Valuation Type = %s'%(lambda_type,vt))
            ax.xaxis.set_label_position('top')  
            ax.set_facecolor('w')
                
    handles, labels = ax.get_legend_handles_labels()   
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels,loc='upper right',frameon =True,framealpha=0.5, ncol=1)
    plt.savefig('Relative_perforamnce.pdf',dpi=600)
    
def plot_auction_allocation_shelby(df_res,ci=None):  
    no_resources = df_res.no_resources.unique().tolist()
    layer = df_res.layer.unique().tolist()
    auction_type = df_res.auction_type.unique().tolist()
    auction_type.remove('')
    valuation_type = df_res.valuation_type.unique().tolist()
    valuation_type.remove('')
    T = len(df_res.t.unique().tolist())

    for idxat,at in enumerate(valuation_type):
        fig, axs = plt.subplots(len(layer), len(no_resources), sharex=True, sharey='col', tight_layout=False)
        for idxnr,nr in enumerate(no_resources):
            for idxvt,vt in enumerate(layer):
                if len(layer)==1 and len(no_resources)==1:
                    ax=axs
                elif len(layer)==1:
                    ax = axs[idxnr]
                elif len(no_resources)==1:
                    ax = axs[idxvt]
                else:
                    ax = axs[idxvt,idxnr]
                    
                with sns.xkcd_palette(['black',"windows blue",'red',"green"]): #sns.color_palette("muted"):
                    ax = sns.lineplot(x='t', y='resource', hue="auction_type", style='decision_type',
                        markers=True, ci=ci, ax=ax,legend='full', 
                        data=df_res[(df_res['layer']==vt)&
                                (df_res['no_resources']==nr)&
                                ((df_res['valuation_type']==at)|(df_res['valuation_type']==''))]) 
                    ax.get_legend().set_visible(False)
                    ax.set(xlabel=r'time step $t$', ylabel='No. resources')
                    ax.xaxis.set_ticks(np.arange(1, 11, 1.0))   #ax.get_xlim()       
#                    ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 1.0), minor=True)
                    ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 1.0))  
                    ax.grid(b=True, which='major', color='w', linewidth=1.0)
#                    ax.grid(b=True, which='minor', color='w', linewidth=0.5)    
                       
        handles, labels = ax.get_legend_handles_labels()
        labels = correct_legend_labels(labels)
        fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5, labelspacing=0.2) #(0.75,0.6)
        
        if len(no_resources)==1 and len(layer)==1:
            axx=[axs]
            axy=[axs]
        elif len(no_resources)==1:
            axx = [axs[0]]
            axy = axs
        elif len(layer)==1:
            axx = axs
            axy = [axs[0]]
        else:
            axx = axs[0,:]
            axy = axs[:,0]  
        fig.suptitle('Valuation Type = '+valuation_type[idxat])
        for idx, ax in enumerate(axx):
            ax.set_title(r'Total resources = %d'%(no_resources[idx]))
        for idx, ax in enumerate(axy):
            ax.annotate('Layer '+`layer[idx]`,xy=(0.1, 0.5),xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',ha='right',va='center',rotation=90)  
  
        plt.savefig('Allocations_'+at+'.pdf',dpi=600)

def plot_auction_allocation_synthetic(df_res,resource_type='resource',ci=None):    
    layer = df_res.layer.unique().tolist()
    auction_type = df_res.auction_type.unique().tolist()
    auction_type.remove('')
    valuation_type = df_res.valuation_type.unique().tolist()
    valuation_type.remove('')
    T = len(df_res.t.unique().tolist())
    
    hor_grid = auction_type
    ver_grid = layer
    new_window = valuation_type
    for idxat,at in enumerate(new_window):
        fig, axs = plt.subplots(len(ver_grid), len(hor_grid), sharex=True, sharey=True, tight_layout=False)
        for idxnr,nr in enumerate(hor_grid):
            for idxvt,vt in enumerate(ver_grid):
                if len(ver_grid)==1 and len(hor_grid)==1:
                    ax=axs
                elif len(ver_grid)==1:
                    ax = axs[idxnr]
                elif len(hor_grid)==1:
                    ax = axs[idxvt]
                else:
                    ax = axs[idxvt,idxnr]
                    
                with sns.xkcd_palette(['black',"windows blue",'red',"green"]): #sns.color_palette("muted"):
                    ax = sns.lineplot(x='t', y=resource_type, hue="decision_type", style='decision_type',
                        markers=True, ci=ci, ax=ax,legend='full', 
                        data=df_res[(df_res['layer']==vt)&
                                ((df_res['auction_type']==nr)|(df_res['auction_type']==''))&
                                ((df_res['valuation_type']==at)|(df_res['valuation_type']==''))]) 
                    ax.get_legend().set_visible(False)
                    ax.set(xlabel=r'time step $t$', ylabel=resource_type)
                    if resource_type=="normalized_resource":
                        ax.set(ylabel=r'\% resource')
                    ax.xaxis.set_ticks(np.arange(1, T+1, 1.0))   #ax.get_xlim()         
                    ax.grid(b=True, which='major', color='w', linewidth=1.0)    
                       
        handles, labels = ax.get_legend_handles_labels()
        labels = correct_legend_labels(labels)
        fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5, labelspacing=0.2) #(0.75,0.6)
        
        if len(hor_grid)==1 and len(ver_grid)==1:
            axx=[axs]
            axy=[axs]
        elif len(hor_grid)==1:
            axx = [axs[0]]
            axy = axs
        elif len(ver_grid)==1:
            axx = axs
            axy = [axs[0]]
        else:
            axx = axs[0,:]
            axy = axs[:,0]  
        fig.suptitle('Valuation Type: '+valuation_type[idxat])
        for idx, ax in enumerate(axx):
            ax.set_title(r'Auction Type: %s'%(hor_grid[idx]))
        for idx, ax in enumerate(axy):
            ax.annotate('Layer '+`int(ver_grid[idx])`,xy=(0.1, 0.5),xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',ha='right',va='center',rotation=90)  
  
        plt.savefig('Allocations_'+at+'.pdf',dpi=600)
        
def plot_relative_allocation_shelby(df_res,distance_type='distance_to_optimal'):   
    no_resources = df_res.no_resources.unique().tolist()
    layer = df_res.layer.unique().tolist()
    decision_type = df_res.decision_type.unique().tolist()
    auction_type = df_res.auction_type.unique().tolist()
    valuation_type = df_res.valuation_type.unique().tolist()
    if '' in valuation_type:
        valuation_type.remove('')
        
#    pals = [sns.color_palette("Blues"),sns.color_palette("Reds"),sns.color_palette("Greens")]
    clrs = [['azure','light blue'],['gold','khaki'],['strawberry','salmon pink'],['green','light green']] #['purple','orchid']
    fig, axs = plt.subplots(len(valuation_type), len(auction_type),sharex=True,
                            sharey=True,tight_layout=False, figsize=(10,7))
    for idxat,at in enumerate(auction_type):   
        for idxvt,vt in enumerate(valuation_type): 
            if len(auction_type)==1 and len(valuation_type)==1:
                ax=axs
            elif len(valuation_type)==1:
                ax = axs[idxat]
            elif len(auction_type)==1:
                ax = axs[idxvt]
            else:
                ax = axs[idxvt,idxat]
            data_ftp = df_res[(df_res['auction_type']==at)&((df_res['valuation_type']==vt)|(df_res['valuation_type']==''))]
            
            for dt in decision_type:
                for nr in no_resources:
                    bottom = 0
                    for P in layer:
                        data_ftp.loc[(data_ftp['layer']==P)&(data_ftp['decision_type']==dt)&(data_ftp['valuation_type']==vt)&(data_ftp['no_resources']==nr),distance_type]+=bottom
                        bottom=data_ftp[(data_ftp['layer']==P)&(data_ftp['decision_type']==dt)&(data_ftp['valuation_type']==vt)&(data_ftp['no_resources']==nr)][distance_type].mean()
                    bottom = 0
                    for P in layer:
                        data_ftp.loc[(data_ftp['layer']==P)&(data_ftp['decision_type']==dt)&(data_ftp['valuation_type']=='')&(data_ftp['no_resources']==nr),distance_type]+=bottom
                        bottom=data_ftp[(data_ftp['layer']==P)&(data_ftp['decision_type']==dt)&(data_ftp['valuation_type']=='')&(data_ftp['no_resources']==nr)][distance_type].mean()

            for P in reversed(layer):    
                with sns.xkcd_palette(clrs[int(P)-1]): #pals[int(P)-1]:
                    ax=sns.barplot(x='no_resources',y='distance_to_optimal',hue="decision_type",
                                data=data_ftp[(data_ftp['layer']==P)], 
                                linewidth=0.5,edgecolor=[.25,.25,.25],
                                capsize=.05,errcolor=[.25,.25,.25],errwidth=.75,ax=ax)
               
            ax.get_legend().set_visible(False)
            ax.grid(which='major', axis='y', color=[.75,.75,.75], linewidth=.75)
            ax.set_xlabel(r'\# resources')
            if idxvt!=len(valuation_type)-1:
                ax.set_xlabel('')
            ax.set_ylabel(r'$E[\omega^k(r^k_d,r^k_c)]$, Valuation: %s' % (valuation_type[idxvt]))
            if idxat!=0:
                ax.set_ylabel('')
            ax.xaxis.set_label_position('bottom')  
            ax.set_facecolor('w')
                
    handles, labels = ax.get_legend_handles_labels()   
    labels = correct_legend_labels(labels)
    for idx,lab in enumerate(labels):
        layer_num = len(layer) - idx//(len(decision_type))
        labels[idx] = lab[:7] + '. (Layer ' + `layer_num` + ')'
    lgd = fig.legend(handles, labels,loc='center', bbox_to_anchor=(0.5, 0.95),
               frameon =True,framealpha=0.5, ncol=4)     #, fontsize='small'
    if len(auction_type)==1 and len(valuation_type)==1:
        axx=[axs]
        axy=[axs]
    elif len(auction_type)==1:
        axx = [axs[0]]
        axy = axs
    elif len(valuation_type)==1:
        axx = axs
        axy = [axs[0]]
    else:
        axx = axs[0,:]
        axy = axs[:,0]
    for idx, ax in enumerate(axx):
        ax.set_title(r'Auction Type: %s'%(auction_type[idx]))
#    for idx, ax in enumerate(axy):
#        rowtitle = ax.annotate('Valuation: '+valuation_type[idx],xy=(0.1, 0.5),xytext=(-ax.yaxis.labelpad - 5, 0),
#            xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center',rotation=90)
    plt.savefig('Allocation_Difference.pdf', bbox_extra_artists=(lgd,), dpi=600)    #, bbox_inches='tight'

def plot_relative_allocation_synthetic(df_res,distance_type='distance_to_optimal'):       
#    no_resources = df_res.no_resources.unique().tolist()
    layer = df_res.layer.unique().tolist()
    decision_type = df_res.decision_type.unique().tolist()
#    decision_type=['judgeCall_OPTIMISTIC']
#    auction_type = df_res.auction_type.unique().tolist()
    valuation_type = df_res.valuation_type.unique().tolist()
    if '' in valuation_type:
        valuation_type.remove('')
    
    hor_grid = [0]
    ver_grid = valuation_type
#    pals = [sns.color_palette("Blues",3),sns.color_palette("Reds",3),sns.color_palette("Greens",3),sns.cubehelix_palette(3)]
    clrs = [['strawberry','salmon pink'],['azure','light blue'],['green','light green'],['bluish purple','orchid']]
    fig, axs = plt.subplots(len(ver_grid), len(hor_grid),sharex=True,sharey=True,tight_layout=False, figsize=(8,6))
    for idxat,at in enumerate(hor_grid):   
        for idxvt,vt in enumerate(ver_grid): 
            if len(hor_grid)==1 and len(ver_grid)==1:
                ax=axs
            elif len(ver_grid)==1:
                ax = axs[idxat]
            elif len(hor_grid)==1:
                ax = axs[idxvt]
            else:
                ax = axs[idxvt,idxat]
            data_ftp = df_res[(df_res['valuation_type']==vt)|(df_res['valuation_type']=='')]
            
            for dt in decision_type:
                bottom = 0
                for P in layer:
                    data_ftp.loc[(data_ftp['layer']==P)&(data_ftp['decision_type']==dt)&(data_ftp['valuation_type']==vt),distance_type]+=bottom
                    bottom=data_ftp[(data_ftp['layer']==P)&(data_ftp['decision_type']==dt)&(data_ftp['valuation_type']==vt)][distance_type].mean()
                bottom = 0
                for P in layer:
                    data_ftp.loc[(data_ftp['layer']==P)&(data_ftp['decision_type']==dt)&(data_ftp['valuation_type']==''),distance_type]+=bottom
                    bottom=data_ftp[(data_ftp['layer']==P)&(data_ftp['decision_type']==dt)&(data_ftp['valuation_type']=='')][distance_type].mean()

            for P in reversed(layer):    
                with sns.xkcd_palette(clrs[int(P)-1]): #pals[int(P)-1]:
                    ax=sns.barplot(x='auction_type',y=distance_type,hue="decision_type",
                                data=data_ftp[(data_ftp['layer']==P)], 
                                linewidth=0.5,edgecolor=[.25,.25,.25],
                                capsize=.05,errcolor=[.25,.25,.25],errwidth=.75,ax=ax)
               
            ax.get_legend().set_visible(False)
            ax.grid(which='major', axis='y', color=[.75,.75,.75], linewidth=.75)
            ax.set_xlabel(r'Auction Type')
            if idxvt!=len(valuation_type)-1:
                ax.set_xlabel('')
            ax.set_ylabel(r'$E[\omega^k(r^k_d,r^k_c)]$, Valuation: %s' %(valuation_type[idxvt]))
            if idxat!=0:
                ax.set_ylabel('')
            ax.xaxis.set_label_position('bottom')  
            ax.set_facecolor('w')
                
    handles, labels = ax.get_legend_handles_labels()   
    labels = correct_legend_labels(labels)
    for idx,lab in enumerate(labels):
        layer_num = len(layer) - idx//(len(decision_type))
        labels[idx] = lab[:7] + '. (Layer ' + `layer_num` + ')'
    lgd = fig.legend(handles, labels,loc='center', bbox_to_anchor=(0.5, 0.95),
               frameon =True,framealpha=0.5, ncol=4)     #, fontsize='small'

    plt.savefig('Allocation_Difference.pdf', bbox_extra_artists=(lgd,), dpi=600)    #, bbox_inches='tight'

def plot_run_time_synthetic(df,ci=None):
    auction_type = df.auction_type.unique().tolist()
    auction_type.remove('')
    decision_type = df.decision_type.unique().tolist()
    valuation_type = df.valuation_type.unique().tolist()
    valuation_type.remove('')
    T = len(df['t'].unique().tolist())
    
    hor_grid=auction_type
    ver_grid=valuation_type
    pals = [sns.color_palette("Blues",3),sns.color_palette("Reds",3),sns.color_palette("Greens",3),sns.cubehelix_palette(3)]
    fig, axs = plt.subplots(len(ver_grid),len(hor_grid),sharex=True,sharey=True,tight_layout=False)
    for idxnr,nr in enumerate(hor_grid):
        for idxvt,vt in enumerate(ver_grid):
            if len(ver_grid)==1 and len(hor_grid)==1:
                ax=axs
            elif len(ver_grid)==1:
                ax = axs[idxnr]
            elif len(hor_grid)==1:
                ax = axs[idxvt]
            else:
                ax = axs[idxvt,idxnr]

            selected_data=df[((df['valuation_type']==vt)|(df['valuation_type']==''))&((df['auction_type']==nr)|(df['auction_type']==''))]
            selected_data['auction_time']=selected_data['auction_time']+selected_data['decision_time']
            selected_data['valuation_time']=selected_data['auction_time']+selected_data['valuation_time']
            value_vars = ['valuation_time','auction_time','decision_time']
            for tt in value_vars: 
                with pals[int(value_vars.index(tt))]:
                    ax=sns.barplot(x='t',y=tt,hue='decision_type',
                                data=selected_data, 
                                linewidth=0.5,edgecolor=[.25,.25,.25],
                                capsize=.05,errcolor=[.25,.25,.25],errwidth=.75,ax=ax)
            ax.set(xlabel=r'time step $t$', ylabel='Run Time (sec)')
            ax.get_legend().set_visible(False)
            ax.xaxis.set_ticks(np.arange(0,T+1,1.0))   #ax.get_xlim()                          
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    for idx,lab in enumerate(labels):
        labels[idx] = value_vars[idx/3]+'('+lab[:7]+')'
    fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5)
    
    if len(ver_grid)==1 and len(hor_grid)==1:
        axx=[axs]
        axy=[axs]
    elif len(hor_grid)==1:
        axx = [axs[0]]
        axy = axs
    elif len(ver_grid)==1:
        axx = axs
        axy = [axs[0]]
    else:
        axx = axs[0,:]
        axy = axs[:,0]  
    for idx, ax in enumerate(axx):
        ax.set_title(r'Auction: %s'%(hor_grid[idx]))
    for idx, ax in enumerate(axy):
        ax.annotate('Valuation: '+ver_grid[idx], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label, textcoords='offset points', ha='right', va='center', rotation=90) 
        
    plt.savefig('Performance_curves.pdf',dpi=600)  

    
def correct_legend_labels(labels):
    labels = ['iINDP' if x=='sample_indp_12Node' else x for x in labels]
    labels = ['JC Optimistic' if x=='sample_judgeCall_12Node_OPTIMISTIC' else x for x in labels]
    labels = ['JC Pessimistic' if x=='sample_judgeCall_12Node_PESSIMISTIC' else x for x in labels]
    labels = ['Auction Type' if x=='auction_type' else x for x in labels]
    labels = ['Decision Type' if x=='decision_type' else x for x in labels]
    labels = ['iINDP' if x=='indp' else x for x in labels]
    labels = ['td-INDP' if x=='tdindp' else x for x in labels]
    labels = ['JC Optimistic' if x=='judgeCall_OPTIMISTIC' else x for x in labels]
    labels = ['JC Pessimistic' if x=='judgeCall_PESSIMISTIC' else x for x in labels]
    labels = ['Auction Type' if x=='auction_type' else x for x in labels]
    labels = ['Decision Type' if x=='decision_type' else x for x in labels]
    labels = ['MCA' if x=='EC' else x for x in labels]
    return labels
