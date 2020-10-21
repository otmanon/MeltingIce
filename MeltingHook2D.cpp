#include <iostream>
#include "MeltingHook2D.h"
#include "imgui/imgui.h"
#include "igl/edges.h"
#include "igl/edge_topology.h"
#include "igl/slice.h"
#include <igl/boundary_facets.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/setdiff.h>
#include <igl/slice.h>
#include <math.h>
#include <igl/slice_into.h>
#include <igl/unique.h>
#include <igl/grad.h>;
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/boundary_loop.h>
#include <igl/winding_number.h>
#include <igl/triangle/triangulate.h>
#include <igl/doublearea.h>
#include <igl/colon.h>
void  MeltingHook2D::initSimulation()
{

	// Load a mesh in OFF format
	//if (modelFilepath.substr(modelFilepath.length() - 3) == ".obj")
	//	igl::readOBJ(modelFilepath, origV, origF);
//	else if (modelFilepath.substr(modelFilepath.length() - 3) == ".off")
//	std::string modelFilepath = "data/2dModels/elephant.off";
	std::string modelFilepath = "data/2dModels/plane.obj";
	Eigen::MatrixXd V;
	Eigen::MatrixXi F, E;
	//igl::readOFF(modelFilepath, V, F);
	igl::readOBJ(modelFilepath, V, F);
	V *= 1.0;
	igl::boundary_facets(F, E); //Fills E with boundary edges... including nonmanifold ones which is what we have in 2D.

	initMesh.V *= 1;
	triangulateDomain(V, F, E);

	segmentDomain(V, E, true);
	D.VertexVel.resize(D.V.rows(), D.V.cols());
	D.VertexVel.setZero();
}

/*
Calculates winding number of each face barycenter. 
Input:
	V - vertices of geometry
	F - faces of geomtry
	interfaceEdges - the interface with respect to which we are calculating winding numbers
*/
void MeltingHook2D::barycenterWindingNumber(Eigen::VectorXd& W, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXi& interfaceEdges)
{
	W.resize(F.rows());
	Eigen::MatrixXd BC;
	// Compute barycenters of all tets
	igl::barycenter(D.V, D.F, BC);
	
	igl::winding_number(V, interfaceEdges, BC, W); //winding number at baricenters

	//normalize
	W = (W.array() - W.minCoeff()) / (W.maxCoeff() - W.minCoeff());
	D.W = W;
	D.BC = BC;
}

void MeltingHook2D::insideOutsideFaces(Eigen::MatrixXi& solidF, Eigen::MatrixXi& liquidF, Eigen::VectorXd& W)
{
	double threshold = 0.9;
	solidF.resize((W.array() > threshold).count(), 3);
	liquidF.resize((W.array() < threshold).count(), 3);
	//update Solid domain and indices 
	int indexS = 0;
	int indexL = 0;
	for (int i = 0; i < D.F.rows(); i++)
	{
		if (D.W(i) > 0.9f) //this point is inside domain
		{
			solidF.row(indexS) = D.F.row(i);
			indexS++;
		}
		else
		{
			liquidF.row(indexL) = D.F.row(i);
			indexL++;
		}
	}
}

/*
returns inVi, outVi, bVi, the indices of the vertices belong to the interior domain, exterior domain and interface domain respectively
Input:
solidF : Flist of index triplets of vertices belonging to interior domain
E	   : Boundary edges, list of doublets containing vertices on the interface's domain.
*/
void insideOutisdeVertices(Eigen::VectorXi& inVi, Eigen::VectorXi& outVi, Eigen::VectorXi& bVi, Eigen::MatrixXi& solidF,int numVertices)
{
	Eigen::VectorXi allIndices, IA, IC, tmp;
	Eigen::MatrixXi interfaceEdges;
	igl::unique(solidF, inVi);														//interiorIndices contains index of interior vertices and includes interface ones too right now
	igl::colon(0, numVertices - 1, allIndices);										//allIndices contains all indeces
	igl::setdiff(allIndices, inVi, outVi, IA);							//exterior indices 
															
	igl::boundary_facets(solidF, interfaceEdges);
	igl::unique(interfaceEdges, bVi);														//get interface indices
	igl::setdiff(inVi, bVi, tmp, IA);						//extracts itnerface indeces from interior vertices and puts them in tmp
	inVi = tmp;
}

void MeltingHook2D::interfaceEdgesToFaceAdjacency(Eigen::MatrixXi& solidF, Eigen::VectorXd & W)
{
	Eigen::MatrixXi interfaceEdges;
	igl::edge_topology(D.V, D.F, D.EV, D.FE, D.EF); //mainly interested in EF
	igl::boundary_facets(solidF, interfaceEdges);

	//identify boundary edges
	D.I.Ei.resize(interfaceEdges.rows());

	D.I.FiL.resize(interfaceEdges.rows());
	D.I.FiS.resize(interfaceEdges.rows());
	int f1, f2;
	int eIndex = 0;
	for (int i = 0; i < D.EV.rows(); i++)
	{
		f1 = D.EF(i, 0);
		f2 = D.EF(i, 1);
		if (f1 == -1.0 || f2 == -1.0)
		{
			continue;
		}
		if (D.W(f1) > 0.9f && D.W(f2) < 0.9f) //this point is inside interface
		{
			D.I.Ei(eIndex) = i; //index of edge in question. (local to global mapping)
			D.I.FiS(eIndex) = f1; //solid f index
			D.I.FiL(eIndex) = f2; //solid f index
			eIndex++;
		}
		else if (D.W(f2) > 0.9f && D.W(f1) < 0.9f)
		{
			D.I.Ei(eIndex) = i; //index of edge in question.
			D.I.FiS(eIndex) = f2; //solid f index
			D.I.FiL(eIndex) = f1; //solid f index
			eIndex++;
		}
	}
	Eigen::VectorXi cols(2, 1);
	cols << 0, 1;
	Eigen::MatrixXi rows;
	igl::slice(D.EV, D.I.Ei, cols, D.I.E);
	D.I.LoopE = interfaceEdges;
}

void MeltingHook2D::interfaceGlobalToLocalVertexMappings()
{
	//Create globalToLocal mappings between interface vertices and global vertices
	int max = D.I.E.maxCoeff();
	D.I.globalToLocalV.resize(max + 1); D.I.globalToLocalV.setConstant(-1);
	D.I.localToGlobalV.resize(D.I.E.rows()); D.I.localToGlobalV.setConstant(-1);
	for (int i = 0; i < D.I.LoopE.rows(); i++)
	{
		if (D.I.globalToLocalV(D.I.LoopE(i, 0)) == -1)
		{
			D.I.globalToLocalV(D.I.LoopE(i, 0)) = i;
			D.I.localToGlobalV(i) = D.I.LoopE(i, 0);
		}
	}

}

void  MeltingHook2D::segmentDomain(Eigen::MatrixXd V, Eigen::MatrixXi E, bool init)
{
	Eigen::VectorXd W; //Winding number of each face
	barycenterWindingNumber(W, V, D.F, E);

	Eigen::MatrixXi solidF, liquidF;
	insideOutsideFaces(solidF, liquidF, W);
	//Count how many vertices are inside Solid and outside Solid
	
	//What's this for? I don't remember.
	Eigen::VectorXd T(D.V.rows());
	
	//label each vertex as being strictly interor, exterior or on the interface
	Eigen::VectorXi interiorIndices, exteriorIndices, interfaceIndices;
	insideOutisdeVertices(interiorIndices, exteriorIndices, interfaceIndices, solidF, D.V.rows());

	//set up interior/exterior Temperatures
	Eigen::VectorXd interiorT = Eigen::VectorXd::Constant(interiorIndices.rows(), 0.0f);
	Eigen::VectorXd interfaceT = Eigen::VectorXd::Constant(interfaceIndices.rows(), 0.0f);
	Eigen::VectorXd exteriorT = Eigen::VectorXd::Constant(exteriorIndices.rows(), 0.0f);

	// maybe unneeded
	//Set bounding box indeces and temperatures
	Eigen::VectorXd  boundaryT;
	
	igl::boundary_facets(D.F, D.Boundary.E);
	igl::unique(D.Boundary.E, D.Boundary.Vi);
	D.Boundary.T = Eigen::VectorXd::Constant(D.Boundary.Vi.rows(), 10);

	//Updates main T with subdomain Ts
	igl::slice_into(interiorT, interiorIndices, 1, T);
	igl::slice_into(exteriorT, exteriorIndices, 1, T);
	igl::slice_into(interfaceT, interfaceIndices, 1, T);
	igl::slice_into(D.Boundary.T, D.Boundary.Vi, T);
	

	//Fill out subdomain info
	D.S.F = solidF;
	D.L.F = liquidF;
	if (init)
	{
		D.S.Vi = interiorIndices;
		D.S.T = interiorT;

		D.L.Vi = exteriorIndices;
		D.L.T = exteriorT;

		D.I.Vi = interfaceIndices;
		
		D.I.T = interfaceT;

		D.T = T;
		
	}

	interfaceEdgesToFaceAdjacency( solidF, W );
	
	interfaceGlobalToLocalVertexMappings();


	//Create globalToLocal mappings between interface vertices and global vertices

}


void  MeltingHook2D::triangulateDomain(Eigen::MatrixXd V, Eigen::MatrixXi F, Eigen::MatrixXi E)
{

	Eigen::VectorXi bi;
	float avgLength = igl::avg_edge_length(V, F);
	// Find boundary vertices
	igl::boundary_facets(F, E); //Fills E with boundary edges... including nonmanifold ones which is what we have in 2D.
	Eigen::VectorXi IA, IC;  //gets indices of vertices on boundary and puts them in b
	igl::unique(E, bi, IA, IC);

	// create Surrounding Bounding Boxe
	Eigen::RowVector3d max = V.colwise().maxCoeff();
	Eigen::RowVector3d min = V.colwise().minCoeff();


	float h = max(1) - min(1);
	float w = max(0) - min(0);

	int NV = V.rows();
	int NE = E.rows();


	V.conservativeResize(V.rows() + 4, 2);
	E.conservativeResize(E.rows() + 4, E.cols());

	D.maxX = max(0) + w * 0.5f;
	D.maxY = max(1) + h * 0.5f;
	D.minX = min(0) - w * 0.5f;
	D.minY = min(1) - h * 0.5f;
	//Add new vertices
	V.row(NV + 0) = Eigen::Vector2d(D.maxX, D.maxY);
	V.row(NV + 1) = Eigen::Vector2d(D.maxX, D.minY);
	V.row(NV + 2) = Eigen::Vector2d(D.minX, D.minY);
	V.row(NV + 3) = Eigen::Vector2d(D.minX, D.maxY);

	// add the last few edges for the bounding box
	E(NE + 0, 0) = NV + 0;	//top right Vert to bottom right
	E(NE + 0, 1) = NV + 1;
	E(NE + 1, 0) = NV + 1;	//bottom right to bottom left
	E(NE + 1, 1) = NV + 2;
	E(NE + 2, 0) = NV + 2;	//bottom left to top left
	E(NE + 2, 1) = NV + 3;
	E(NE + 3, 0) = NV + 3;	//top left to top right
	E(NE + 3, 1) = NV + 0;


	//horizontal edges(easy)
	Eigen::MatrixXd V2D, V2;
	Eigen::MatrixXi F2;
	//	convert3DVerticesTo2D(V, V2D);
	Eigen::MatrixXd H;
	igl::triangle::triangulate(V, E, H, "a1.0q", V2, F2);
	convert2DVerticesTo3D(V2, V);
	D.V = V;
	D.F = F2;
	allV = D.V;
	D.F;

}


void  MeltingHook2D::updateDomain()
{
	//Save previous version of mesh
	Eigen::MatrixXd V = D.V;
	Eigen::MatrixXi F = D.F;

	//re triangulate domain
	retriangulateDomain();
	//re segment domain
	segmentDomain(V, D.I.LoopE, false);

}


void  MeltingHook2D::retriangulateDomain()
{

	Eigen::MatrixXi boundaryEdges;

	boundaryEdges.resize(D.Boundary.E.rows() + D.I.E.rows(), D.I.E.cols());
	boundaryEdges << D.Boundary.E, D.I.E;
	//horizontal edges(easy)
	Eigen::MatrixXd V2D, V2;
	Eigen::MatrixXi F2;
	convert3DVerticesTo2D(D.V, V2D);
	Eigen::MatrixXd H;
	igl::triangle::triangulate(V2D, boundaryEdges, H, "a0.25qYY", V2, F2);
	convert2DVerticesTo3D(V2, D.V);
	D.F = F2;
	//	D.VGrad = Eigen::MatrixXd::Zero(D.V.rows(), D.V.cols());
	//	D.VN = Eigen::MatrixXd::Zero(D.V.rows(), D.V.cols());
}


void MeltingHook2D::drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu)
{
	ImGui::Checkbox("stepping mode", &stepping_mode);
	ImGui::Checkbox("step sim", &stepping_flag);

	
	
	
	if (ImGui::CollapsingHeader("Rendering"))
	{
		ImGui::Checkbox("render Fluid", &renderFluid);
		ImGui::Checkbox("render Solid", &renderSolid);
		ImGui::Checkbox("render Global", &renderGlobal);
		ImGui::Checkbox("render Normals", &renderNormals);
		ImGui::Checkbox("render Temp Gradient", &renderTGrad);
		ImGui::Checkbox("render Vp", &renderVp);
		ImGui::Checkbox("render VertexVel", &renderVertexVel);
		ImGui::Checkbox("Render Interpolated", &renderInterpolatedVel);
		ImGui::SliderFloat("vis scale", &vis_scale, 1e-2, 1e2, "%.8f", 10.0f);
	}

	if (ImGui::CollapsingHeader("Sim. Parameters"))
	{
		ImGui::Checkbox("do diffusion", &diffusion_flag);
		ImGui::Checkbox("do melting", &melting_flag);
		ImGui::Checkbox("re-triangulate", &retriangulate);
		ImGui::SliderFloat("dt", &dt, 1e-6, 1e1, "%.5f", 10.0f);
		ImGui::SliderFloat("lambda", &D.lambda, 0, 1e2, "%.2f", 10.0f);
		ImGui::SliderFloat("phi", &phi, 0, 100);

	}
	


	
}


void  MeltingHook2D::convert3DVerticesTo2D(Eigen::MatrixXd& v3D, Eigen::MatrixXd& v2D)
{
	v2D.resize(v3D.rows(), 2);// = Eigen::MatrixXd::Zero(solid.V.rows(), 2);

	//Fill 2D V
	for (int i = 0; i < v2D.rows(); i++)
	{
		v2D(i, 0) = v3D(i, 0);
		v2D(i, 1) = v3D(i, 1);
	}
}


void  MeltingHook2D::convert2DVerticesTo3D(Eigen::MatrixXd& v2D, Eigen::MatrixXd& v3D)
{
	v3D.resize(v2D.rows(), 3);// = Eigen::MatrixXd::Zero(solid.V.rows(), 2);

	//Fill 2D V
	for (int i = 0; i < v3D.rows(); i++)
	{
		v3D(i, 0) = v2D(i, 0);
		v3D(i, 1) = v2D(i, 1);
		v3D(i, 2) = 0;
	}
}


bool  MeltingHook2D::simulateOneStep()
{
	
	if (stepping_mode)
	{
		if (stepping_flag)
		{
			if (explicitMelting)
			{
				explicitStep(dt, 1000.0);
				stepping_flag = false;

			}
		}
	}
	else
	{
		if (explicitMelting)
		{
			explicitStep(dt, 1000.0);
		}
		else if (implicitMelting)
		{

		}


	}

	return false;
}

void MeltingHook2D::explicitStep(float dt, float latentHeat)
{
	if (diffusion_flag)
	{
		D.diffuseHeat(dt);
		D.calculateQuantities(1000.0);
		if (renderInterpolatedVel)
			D.calculateInterpolationAlongEdges();
		
	}
	
	if (melting_flag)
	{
		D.melt(dt);

	}
	if (retriangulate)
	{
		updateDomain();//once we melt we need to make sure our geometry is still nice. 
	}
}

void MeltingHook2D::implicitStep(float dt, float latentHeat)
{
	/*
	if (diffusion_flag)
	{
		D.diffuseEnthalpy(dt);
	}
	
	if (extract_isocontour)
	{
		D.diffuseEnthalpy(dt);
	}*/
}

void  MeltingHook2D::updateRenderGeometry()
{
	//V and T always the same
	renderV = D.V;
	renderT = D.T;

	if (renderFluid && !renderSolid)
		renderF = D.L.F;
	else if (renderSolid && !renderFluid)
		renderF = D.S.F;
	else
	{
		renderF = D.F;
	}
}


void MeltingHook2D::renderRenderGeometry(igl::opengl::glfw::Viewer &viewer)
{
	//	viewer.data().clear_edges();
	viewer.data().clear();
	//	viewer.data().clear_edges();
	viewer.data().set_mesh(renderV, renderF);
	viewer.data().set_data(renderT, 0, 10);

	if (renderTGrad)
	{
		const Eigen::RowVector3d green(0, 1, 0);
		viewer.data().add_edges(D.BC, D.BC + vis_scale * D.FGrad, green);

	}
	if (renderNormals)
	{
		const Eigen::RowVector3d red(1, 0, 0);
		if(!renderGlobal)
			viewer.data().add_edges(D.I.MidPE, D.I.MidPE + vis_scale * D.I.NormalsE, red);
		else
			viewer.data().add_edges(D.MidPE, D.MidPE + vis_scale * D.NormalsE, red);
	}
	if (renderVp)
	{
		const Eigen::RowVector3d white(1, 1, 1);
		if (!renderGlobal)
			viewer.data().add_edges(D.I.MidPE, D.I.MidPE + vis_scale * D.I.Vp, white);
		else
			viewer.data().add_edges(D.MidPE, D.MidPE + vis_scale * D.Vp, white);
	}
	if (renderVertexVel)
	{
		const Eigen::RowVector3d white(1, 1, 1);
		viewer.data().add_edges(D.V, D.V + vis_scale * D.VertexVel, white);

	}
	if (renderInterpolatedVel)
	{
		const Eigen::RowVector3d red(1, 0, 0);
		viewer.data().add_edges(D.I.intX, D.I.intX + vis_scale * D.I.intV, red);
	}
}