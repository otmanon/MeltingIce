#include "PhysicsHook.h"
#include <igl/boundary_facets.h>
#include <igl/colon.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/jet.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/setdiff.h>
#include <igl/slice.h>
#include <math.h>
#include <igl/slice_into.h>
#include <igl/unique.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Sparse>
#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <igl/grad.h>;
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/boundary_loop.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/winding_number.h>
#include <igl/triangle/triangulate.h>

struct Domain
{
	Eigen::MatrixXd origV; //original mesh description. Remember it when resetting
	Eigen::MatrixXi origF;

	Eigen::MatrixXd V;	//mesh geometry for solid
	Eigen::MatrixXi F;
	Eigen::MatrixXi E;  //Matrix of edges (N, 2), where each row is an edge, whose entries are both vertex indices
	
	Eigen::MatrixXd FGrad; //Gradient on faces 
	Eigen::MatrixXd VGrad; //Gradient on vertices coming from solid(average of nearby faces) 

	Eigen::VectorXd T; // Temperature at each vertex
	Eigen::VectorXd Tb;	// temperature at each boundary vertex
	Eigen::VectorXi interfaceIndices; // indices of vertices on the interface
	Eigen::VectorXi boundaryIndices; // indices of vertices at mesh boundary (for solid it's the same as interfaceIndices, for liquid, it includes interface Indeces as well as border

	Eigen::MatrixXd EM; //Edge Midpoints
	Eigen::MatrixXd EN; //Edge Normals
	Eigen::MatrixXd VN; //2D normal stored at each vertex by averadging incident edge normals
	Eigen::VectorXd W; //Winding number...

	/*
	Diffuses heat by solving the heat equation by minmizing a semi-implicit time update of the Energy Functional :
	(u_n+1)^T (I - dt * C)u_n+1 -  (u_n+1)^T(u_n)

	Inputs:
	V: Nx3 Matrix of vertices
	F: Mx3 Matrix of faces
	bIndices : Vector containing indices of boundary
	TBoundary: Vector containing boundary conditions at each boundary vertex

	Output:
	T : Temperature at the current timestep, which will be overwritten as the quadratic energy min. takes effect.
	*/
	void diffuseHeat(float dt)
	{
		Eigen::SparseMatrix<double> C, I, Q; //cotan matrix, identity matrix, quadratic coefficient term
		Eigen::VectorXd B = Eigen::VectorXd::Zero(V.rows(), 1); // linear coefficients
		B = -T; //Linear coefficients are equal to last timesteps solved Temperature field

		igl::cotmatrix(V, F, C);
		I.resize(V.rows(), V.rows());
		I.setIdentity();
		Q = I - dt * C;
		igl::min_quad_with_fixed_data<double> mqwf;
		Eigen::VectorXd Beq;
		Eigen::SparseMatrix<double> Aeq;
		igl::min_quad_with_fixed_precompute(Q, boundaryIndices, Aeq, false, mqwf);
		igl::min_quad_with_fixed_solve(mqwf, B, Tb, Beq, T);

	}
};

struct SubDomain
{
	Domain parentDomain;
	Eigen::MatrixXi F; // Faces belonging to that domain. The indices in the faces refer to vertices in the parent Domains V matrix
	Eigen::VectorXd T; // Temperatures associated to the vertices of this domain
	Eigen::VectorXd Vi;// Indices from parentDomain's Vertices list that belongs to this domain

};

class MeltingHook2D : public PhysicsHook
{
public:
	MeltingHook2D() : PhysicsHook() {}

	/*
	Initializes simulation. 
	*/
	virtual void initSimulation()
	{
	
		// Load a mesh in OFF format
		//if (modelFilepath.substr(modelFilepath.length() - 3) == ".obj")
		//	igl::readOBJ(modelFilepath, origV, origF);
	//	else if (modelFilepath.substr(modelFilepath.length() - 3) == ".off")
			igl::readOFF(modelFilepath, solid.V, solid.F);
			igl::boundary_facets(solid.F, solid.E); //Fills E with boundary edges... including nonmanifold ones which is what we have in 2D.

// Find boundary vertices
		//	Eigen::VectorXi IA, IC;  //gets indices of vertices on boundary and puts them in b
		//	igl::unique(solid.E, solid.interfaceIndices, IA, IC);

		solid.V *= 2;

		triangulateDomain();
		
		segmentDomain();
		


		solid.VGrad = Eigen::MatrixXd::Zero(solid.V.rows(), solid.V.cols());
		liquid.VGrad = Eigen::MatrixXd::Zero(liquid.V.rows(), liquid.V.cols());



		
	}

	/*
	Divides the domain into solid/liquid/boundary
	*/
	void segmentDomain()
	{
		Eigen::VectorXd W;
		Eigen::MatrixXd BC;
		// Compute barycenters of all tets
		igl::barycenter(domain.V, domain.F , BC);
		igl::winding_number(solid.V, solid.E, BC, domain.W); //winding number at baricenters
		//normalize
		domain.W = (domain.W.array() - domain.W.minCoeff()) / (domain.W.maxCoeff() - domain.W.minCoeff());
		W = domain.W;

		//Count how many vertices are inside Solid and outside Solid
		Eigen::MatrixXi solidF((W.array() > 0.9f).count(), 3 ); // faces inside solid
		Eigen::MatrixXi liquidF((W.array() < 0.9f).count(), 3); // faces inside solid
		Eigen::VectorXd T(domain.V.rows());
		//update Solid domain and indices 
		int indexS = 0;
		int indexL = 0;
		for (int i = 0; i < domain.F.rows(); i++)
		{
			if (domain.W(i) > 0.9f) //this point is inside domain
			{
				solidF.row(indexS) = domain.F.row(i);
				indexS++;
			}
			else
			{
				liquidF.row(indexL) = domain.F.row(i);
				indexL++;
			}
		}

		Eigen::MatrixXd interfaceEdges;
		Eigen::VectorXi interiorIndices, interiorIndices2, allIndices, exteriorIndices, interfaceIndices, IA, IC;
		igl::unique(solidF, interiorIndices); //interiorIndices contains index of interior vertices
		igl::colon(0, domain.V.rows() - 1, allIndices); //allIndices contains all indeces
		igl::setdiff(allIndices, interiorIndices, exteriorIndices, IA); //exterior indices 
		igl::boundary_facets(solidF, interfaceEdges); //get edges of solid
		igl::unique(interfaceEdges, interfaceIndices); //get interface indices
		//add interfaceIndices to exteriorIndices

		//separate  boundary indices from interior indices.
		igl::setdiff(interiorIndices, interfaceIndices, interiorIndices2, IA);
		interiorIndices = interiorIndices2;

		//set up interior/exterior Temperatures
		Eigen::VectorXd interiorT = Eigen::VectorXd::Constant(interiorIndices.rows(), 0.0f);
		Eigen::VectorXd exteriorT = Eigen::VectorXd::Constant(exteriorIndices.rows(), 10.0f);
		igl::slice_into(interiorT, interiorIndices , 1, T);
		igl::slice_into(exteriorT, exteriorIndices, 1, T);

		domain.T = T;
	//	Eigen::MatrixXi E; //boundary of
	//	igl::boundary_facets(solidF, E);
		//gets indices of vertices on boundary and puts them in b
	//	igl::unique(solid.E, solid.interfaceIndices, IA, IC);
		//update Liquid domain and indices


	}

	/*
	Given solid mesh, extracts its boundary, wraps a bounding box around it, triangulates 
	entire domain in one sweep, while maintaining elephant boundary edges
	*/
	void triangulateDomain()
	{
		Eigen::MatrixXd V = solid.V;
		Eigen::MatrixXi F = solid.F;
		Eigen::MatrixXi E = solid.E;
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


		V.conservativeResize( V.rows() + 4, 2);
		E.conservativeResize(E.rows() + 4, E.cols());

		//Add new vertices
		V.row(NV + 0) = Eigen::Vector2d(max(0) + w * 0.5f, max(1) + h * 0.5f);
		V.row(NV + 1) = Eigen::Vector2d(max(0) + w * 0.5f, min(1) - h * 0.5f);
		V.row(NV + 2) = Eigen::Vector2d(min(0) - w * 0.5f, min(1) - h * 0.5f);
		V.row(NV + 3) = Eigen::Vector2d(min(0) - w * 0.5f, max(1) + h * 0.5f);

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
		igl::triangle::triangulate(V, E, H, "a0.005q", V2, F2);
		convert2DVerticesTo3D(V2, V);
		domain.V = V;
		domain.F = F2;
		allV = domain.V;
		domain.F;


	}


	/*
	ImGui render... not sure how this works
	*/
	virtual void drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu)
	{
		ImGui::Checkbox("stepping mode", &stepping_mode);
		ImGui::Checkbox("step sim", &stepping_flag);

		
		ImGui::Checkbox("render Fluid", &renderFluid);
		ImGui::Checkbox("render Solid", &renderSolid);


		ImGui::Checkbox("do diffusion", &diffusion_flag);
		ImGui::Checkbox("do melting", &melting_flag);
		ImGui::SliderFloat("dt", &dt, 1e-6, 1e1, "%.5f", 10.0f);
		ImGui::SliderFloat("phi", &phi, 0, 100);
		ImGui::SliderFloat("vis scale", &vis_scale, 1e-4, 1e-1, "%.8f", 10.0f);
	}


	/*
	Given gradient of scalar field accross each vertex, calculates the gradient at each vertex (average from neighborin gfaces)
	*/
	void calculateVertexGradientSolid(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& FGrad)
	{
		solid.VGrad = Eigen::MatrixXd::Zero(V.rows(), 3);
		Eigen::VectorXd VGradWeight = Eigen::VectorXd::Zero(V.rows()); //keeps track of number of faces that have contributed to vertex
		for (int i = 0; i < FGrad.rows(); i++) //loop through faces, distribute gradients to each vertex
		{
			int v1_index = F(i, 0), v2_index = F(i, 1), v3_index = F(i, 2);

			solid.VGrad.row(v1_index) += FGrad.row(i);
			VGradWeight(v1_index) += 1;

			solid.VGrad.row(v2_index) += FGrad.row(i);
			VGradWeight(v2_index) += 1;

			solid.VGrad.row(v3_index) += FGrad.row(i);
			VGradWeight(v3_index) += 1;

		}

		for (int i = 0; i < solid.VGrad.rows(); i++)
		{

			solid.VGrad.row(i) /= VGradWeight(i);
		}

	}

	/*
	Calculates the gradient of the temperature field near the surface. Assumes gradient is constant on the outside.
	Then, moves the surface inwards based on the stefan condition:
	Lv = ks dTs/dn - kl dTl/dn
	*/
	void meltSurface()
	{

		Eigen::MatrixXd DTDN = Eigen::MatrixXd::Zero(solid.VN.rows(), solid.VN.cols());
		// For each vertex, get the normal and the gradient, then dot the normal against the gradient
		for (int i = 0; i < solid.V.rows(); i++)
		{
			Eigen::Vector3d normal = solid.VN.row(i);
			Eigen::Vector3d grad = - ( liquid.VGrad.row(i) + solid.VGrad.row(i));
			Eigen::Vector3d dTdn = normal.dot(grad) * normal;
			DTDN.row(i) = dTdn;
		}
		solid.V = solid.V - DTDN * dt;
	}

	/*
	Calculates all info having to do with gradient of temperature field
	*/
	void calculateGradientInfo()
	{
		Eigen::SparseMatrix<double> G;  //Gradient operator. Takes scalar (T) stored at each vertex, returns gradient of T at each face.
		igl::grad(solid.V, solid.F, G);	//Get gradient operator

		solid.FGrad = Eigen::Map<const Eigen::MatrixXd>((G*solid.T).eval().data(), solid.F.rows(), 3); //Gradient of T on each face.

		calculateVertexGradientSolid(solid.V, solid.F, solid.FGrad); //Go through each face and distribute gradient to face's vertices.


		calculateBoundaryMidpointsAndNormals();
		//Eigen::MatrixXd VGradExt = getVertexGradientExt(V, F, VN);
		calculateVertexGradientLiquid();
	}

	/*
	Loops through all boundary edges. Deposits their contributions to their boundary vertices. 
	*/
	void calculateVertexGradientLiquid()
	{
		Eigen::VectorXi bndEdges;
		igl::boundary_loop(solid.F, bndEdges);
		//Assuming boundary is closed loop, each vertex has 2 edges. 
		liquid.VGrad = Eigen::MatrixXd::Zero(solid.V.rows(), 3);
		
		for (int i = 0; i < bndEdges.rows(); i++) //loop through faces, distribute gradients to each vertex
		{
			int i1 = i;
			int i2 = (i + 1) % bndEdges.rows(); //Mod operators bc last index is attached to first index

			liquid.VGrad.row(bndEdges(i1)) += -0.5f *  phi * solid.VN.row(bndEdges(i1));
			liquid.VGrad.row(bndEdges(i2)) += -0.5f * phi * solid.VN.row(bndEdges(i2));

		}
	}
	/*
	Get boundary vertices and normals
	Takes as input the vertices and faces of the mesh, V and F
	Outputs :
	EM, the modpoint of each boundary edge
	EN the normal at each boundary edge
	VN the normal at each boundary vertex
	*/
	void calculateBoundaryMidpointsAndNormals()
	{
		Eigen::VectorXi bnd;
		igl::boundary_loop(solid.F, bnd);
		solid.EM.resize(bnd.rows(), 3); //M holds the midpoints for each boundary edge. x, y, and z coords
		solid.EN.resize(bnd.rows(), 3); //N holds the normals for each boundary edge. x, y, and z coords
		solid.VN.resize(solid.V.rows(), 3); //VN holds the normals for each vertex
		solid.VN.setZero();

		for (int i = 0; i < bnd.rows(); i++)
		{
			int i1 = i;
			int i2 = (i + 1) % bnd.rows(); //Mod operators bc last index is attached to first index

			Eigen::Vector3d midpoint = (solid.V.row(bnd(i2)) + solid.V.row(bnd(i1))) / 2;
			Eigen::Vector3d r = (solid.V.row(bnd(i2)) - solid.V.row(bnd(i1)));
			Eigen::Vector3d z(0, 0, 1);
			Eigen::Vector3d normal = r.cross(z);
			normal.normalize();
			solid.EM.row(i) = midpoint;
			solid.EN.row(i) = normal;

			//Now deposit the edge normal towards the vertex normal
			solid.VN.row(bnd(i1)) += solid.EN.row(i);
			solid.VN.row(bnd(i2)) += solid.EN.row(i);

		}

		for (int i = 0; i < solid.VN.rows(); i++)
		{
			solid.VN.row(i).normalize();
		}
	};

	/*
	Converts 3D vertices (used in libigl) to 2D (used in triangle)
	*/
	void convert3DVerticesTo2D(Eigen::MatrixXd& v3D, Eigen::MatrixXd& v2D)
	{
		v2D.resize(v3D.rows(), 2);// = Eigen::MatrixXd::Zero(solid.V.rows(), 2);

		//Fill 2D V
		for (int i = 0; i < v2D.rows(); i++)
		{
			v2D(i, 0) = v3D(i, 0);
			v2D(i, 1) = v3D(i, 1);
		}
	}

	/*
	converts 2D vertices to 3D 
	*/
	void convert2DVerticesTo3D( Eigen::MatrixXd& v2D, Eigen::MatrixXd& v3D)
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

	/*
	Simulates one timestep
	*/
	virtual bool simulateOneStep()
	{
		if (stepping_mode)
		{
			if (stepping_flag)
			{
				if (diffusion_flag)
				{
					liquid.diffuseHeat(dt);
					calculateGradientInfo();
				}
				if(melting_flag)
				//	meltSurface();
				stepping_flag = false;
			}
		}
		else
		{
			liquid.diffuseHeat(dt);
			calculateGradientInfo();
			//meltSurface();
		}

		return false;
	}


	/*

	*/
	virtual void updateRenderGeometry()
	{
	//	if (renderFluid)
	//	{
	//		renderV = liquid.V;
	///		renderF = liquid.F;
	///		renderT = liquid.T;
	//	}
	///	else if (renderSolid)
	//	{
		//	renderV = solid.V;
		//	renderF = solid.F;
	//		renderT = solid.T;
	//	}
	//	else {
		renderV = domain.V;
		renderF = domain.F;
		renderT = domain.T;
	///	}


	}


	virtual void renderRenderGeometry(igl::opengl::glfw::Viewer &viewer)
	{
		viewer.data().clear();
	//	viewer.data().clear_edges();
		viewer.data().set_mesh(renderV, renderF);
		viewer.data().set_data(renderT, 0, 10);

	//	const Eigen::RowVector3d black(0, 0, 0);
	//	viewer.data().add_edges(solid.V, solid.V + vis_scale * solid.VGrad, black);

	//	const Eigen::RowVector3d red(1, 0, 0);
	//	viewer.data().add_edges(solid.V, solid.V + vis_scale * solid.VGrad, red);
	}

private:
	bool renderFluid = true;
	bool renderSolid = false;

	bool stepping_mode = false;
	bool stepping_flag = false;
	bool melting_flag = true;
	bool diffusion_flag = true;
	float vis_scale = 0.0001;
	float phi = 50; //flux applied at each edge
	float dt = 1e-1; //timestep

	Eigen::MatrixXd origV; //original mesh description. Remember it when resetting
	Eigen::MatrixXi origF;

	Eigen::MatrixXd allV; // includes Solid geometry and liquid geometry.
	Eigen::MatrixXi allF; 
	Eigen::VectorXd allT;

	Eigen::MatrixXd renderV; // Mesh description that will be rendered (ie either solid geometry, liquid geometry, or both
	Eigen::MatrixXi renderF;
	Eigen::VectorXd renderT;

	std::string modelFilepath = "data/2dModels/elephant.off";

	Domain solid;
	Domain liquid;
	Domain domain;
};