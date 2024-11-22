
import plotly.graph_objects as go
import numpy as np
from aether.aether_mesh import Mesh

class MeshExplorer:
    
    def __init__(self, mesh : Mesh, title='Mesh Explorer'):
        self.mesh = mesh
        self.fig = go.Figure()
        
        self.fig.update_layout(
            title=title,
            xaxis=dict(scaleanchor="y", showgrid=False, zeroline=True),
            yaxis=dict(showgrid=False, zeroline=True),
            width=1000,
            height=1000,
            showlegend=False,
        )

        
    def add_edges(self):
        """
        Add edge trace.
        """
        mesh = self.mesh 
        fig = self.fig
        edges = mesh.edge_to_vertices
        coordinates = mesh.coordinates 
        edges = mesh.edge_to_vertices
        
        c0 = coordinates[edges[:,0]]
        c1 = coordinates[edges[:,1]]
        x = np.c_[c0[:,0], c1[:,0], [None]*len(c0)].flatten()
        y = np.c_[c0[:,1], c1[:,1], [None]*len(c0)].flatten()

        fig = self.fig 
        index = len(fig.data)
        self.edge_trace_index = index 
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color='black'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
            
              
    def add_vertices(self):
        """
        Add vertex trace.
        """
        
        mesh = self.mesh 
        fig = self.fig
        coordinates = mesh.coordinates
    
        index = len(fig.data)
        self.vertex_trace_index = index 
        
        fig.add_trace(
            go.Scatter(
                x=coordinates[:,0],
                y=coordinates[:,1],
                mode='markers',
                hoverinfo='text',
                hovertemplate =
                '<b>x</b>: %{x:.2f}'+
                '<br><b>y</b>: %{y:.2f}<br>'+
                '<b>%{text}</b>',
                text = [f'Vertex: {i}' for i in range(mesh.num_vertices)],
                marker=dict(size=10, color='rgba(0,0,0,1)'),
                customdata=['vertex']*mesh.num_vertices
            )
        )


   