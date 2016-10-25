#pragma once

#include <type_traits>
#include <initializer_list>

#include "coredefs.h"
#include "../ndarray/intn.h"

///////////////////Sasha: proposing switch from Size<n> to intn<n> implementation //////////////
namespace iu{
	namespace depricated{
		template<unsigned int dims> class Size;
	};
	template<int dims>
	// proposed implementation
	using Size = ::intn<dims>; // replacement of Size implementation
	// old implementation
	//using Size = to_be_depricated::Size<dims>; // replacement of Size implementation
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace iu{
	/** \brief Base class for N-dimensional vectors. */
	template<typename PixelType, unsigned int Ndim>
	class VectorBase
	{
		IU_ASSERT(Ndim > 0)

	public:
		/** Constructor. */
		VectorBase()
		{
			for (unsigned int i = 0; i < Ndim; i++)
			{
				data_[i] = static_cast<PixelType>(0.0);
			}
		}

		/** Special Constructor.
		 *  Init all elements of the vector with a special value.
		 *  @param value value to initialize vector elements.*/
		explicit VectorBase(const PixelType& value)
		{
			this->fill(value);
		}

		/** Special Constructor.
		 *  Init all elements of the vector with a initializer list.
		 *  @param list Initializer list, e.g. {1,2,3}.*/
		VectorBase(std::initializer_list<PixelType> list)
		{
			if (list.size() != Ndim)
			{
				std::stringstream msg;
				msg << "Length of initializer list (" << list.size();
				msg << ") does not match number of size dimensions (" << Ndim << ").";
				throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
			}

			unsigned int i = 0;
			for (auto elem : list)
			{
				data_[i] = elem;
				++i;
			}
		}

		//! construct from variadic list -- allows implicit element type conversion (e.g .sfrom short, unsigned, etc.)
		/*!
		 * example: Size<3>(12, 2, 3);
		 */
		/* Size specializations fail to inherit this
		template<typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>::value>::type>
		VectorBase(A0 a0, Args... args) : VectorBase(std::initializer_list<int>( {int(a0), int(args)...} )){
			static_assert(sizeof...(Args)==Ndim-1,"size missmatch"); // check number of arguments is matching
		}
		*/

		/** Constructor from other array type, can be C array, std::vector, etc.
		 *
		 */
		template<class other, class = typename std::enable_if<std::is_class<other>::value>::type>
		explicit VectorBase(const other & x){
			for(unsigned int i=0; i< Ndim; ++i){
				(*this)[i] = x[i];
			};
		}

		/** Destructor. */
		virtual ~VectorBase()
		{
		}

		/** Overload [] to access the size elements. */
		inline PixelType operator[](unsigned int i) const
		{
			return data_[i];
		}

		/** Overload [] to access the size elements. */
		inline PixelType &operator[](unsigned int i)
		{
			return data_[i];
		}

		/** Get number of dimensions. */
		static unsigned int ndim()
		{
			return Ndim;
		}

		/** Fill the vector with a specific value. */
		void fill(const PixelType& value)
		{
			for (unsigned int i = 0; i < Ndim; i++)
			{
				data_[i] = value;
			}
		}

		/** Operator<< overloading for Ndim==1. Output of VectorBase class. */
		template<typename T = std::ostream>
		friend typename std::enable_if<(Ndim == 1), T&>::type operator<<(
				std::ostream & out, VectorBase<PixelType, Ndim> const& v)
		{
			out << "[" << v[0] << "]";
			return out;
		}

		/** Operator<< overloading for Ndim>1. Output of VectorBase class. */
		template<typename T = std::ostream>
		friend typename std::enable_if<(Ndim > 1), T&>::type operator<<(
				std::ostream & out, VectorBase<PixelType, Ndim> const& v)
		{
			out << "[";
			for (int i = 0; i < static_cast<int>(Ndim - 1); i++)
			{
				out << v[i] << ", ";
			}
			out << v[Ndim - 1] << "]";
			return out;
		}

	protected:
		/** data buffer */
		PixelType data_[Ndim];

	private:
		/** Private copy constructor. */
		VectorBase(const VectorBase& from);
		/** Private copy assignment operator. */
		VectorBase& operator=(const VectorBase& from);
	};

	////////////////////////////////////////////////////////////////////////////////
	/** Check two vectors for equality. */
	template<typename PixelType, unsigned int Ndim>
	bool operator==(const VectorBase<PixelType, Ndim> &v1,
			const VectorBase<PixelType, Ndim> &v2)
			{
		for (unsigned int i = 0; i < Ndim; i++)
		{
			if (v1[i] != v2[i])
				return false;
		}
		return true;
			}

	/** Check two vectors for inequality. */
	template<typename PixelType, unsigned int Ndim>
	bool operator!=(const VectorBase<PixelType, Ndim> &v1,
			const VectorBase<PixelType, Ndim> &v2)
			{
		for (unsigned int i = 0; i < Ndim; i++)
		{
			if (v1[i] != v2[i])
				return true;
		}
		return false;
			}

	////////////////////////////////////////////////////////////////////////////////
	/** \brief Main Class for N-dimensional vectors.
	 *
	 * This class defines also some useful operator overloading
	 */
	template<typename PixelType, unsigned int Ndim>
	class Vector: public VectorBase<PixelType, Ndim>
	{
	public:
		/** Constructor */
		Vector() = default;
		//using VectorBase<PixelType, Ndim>::VectorBase; // inheriting all constructors of VectorBase
		//using VectorBase<unsigned int, Ndim>::VectorBase;
		//! forward constructor to VectorBase
		template <typename A0, typename... Args>
		Vector(A0 && a0, Args&&... args) : VectorBase<unsigned int, Ndim>(std::forward<A0>(a0), std::forward<Args>(args)...){
		}


		//  Vector() :
		//      VectorBase<PixelType, Ndim>()
		//  {
		//  }
		//
		//  /** Special Constructor.
		//   *  Init all elements of the vector with a special value.
		//   *  @param value value to initialize vector elements.*/
		//  Vector(const PixelType& value) :
		//      VectorBase<PixelType, Ndim>(value)
		//  {
		//  }
		//
		//  /** Special Constructor.
		//   *  Init all elements of the vector with a initializer list.
		//   *  @param list Initializer list, e.g. {1,2,3}.*/
		//  Vector(std::initializer_list<PixelType> list) :
		//      VectorBase<PixelType, Ndim>(list)
		//  {
		//
		//  }

		/** Destructor. */
		~Vector()
		{
		}

		/** Overload operator+. Add scalar. */
		inline Vector<PixelType, Ndim> operator+(const PixelType &scalar) const
		{
			Vector<PixelType, Ndim> v;
			for (unsigned int i = 0; i < Ndim; i++)
			{
				v[i] = this->data_[i] + scalar;
			}
			return v;
		}

		/** Overload operator+. Add vector. */
		inline Vector<PixelType, Ndim> operator+(Vector<PixelType, Ndim> &v1) const
		{
			Vector<PixelType, Ndim> v2;
			for (unsigned int i = 0; i < Ndim; i++)
			{
				v2[i] = this->data_[i] + v1[i];
			}
			return v2;
		}

		/** Overload operator+. Subtract scalar. */
		inline Vector<PixelType, Ndim> operator-(const PixelType &scalar) const
		{
			Vector<PixelType, Ndim> v;
			for (unsigned int i = 0; i < Ndim; i++)
			{
				v[i] = this->data_[i] - scalar;
			}
			return v;
		}

		/** Overload operator+. Subtract vector. */
		inline Vector<PixelType, Ndim> operator-(Vector<PixelType, Ndim> &v1) const
		{
			Vector<PixelType, Ndim> v2;
			for (unsigned int i = 0; i < Ndim; i++)
			{
				v2[i] = this->data_[i] - v1[i];
			}
			return v2;
		}

		/** Overload operator*. Multiplication with scalar. */
		inline Vector<PixelType, Ndim> operator*(const PixelType &scalar) const
		{
			Vector<PixelType, Ndim> v;
			for (unsigned int i = 0; i < Ndim; i++)
			{
				v[i] = this->data_[i] * scalar;
			}
			return v;
		}

		/** Overload operator*. Multiplication with vector. */
		inline Vector<PixelType, Ndim> operator*(Vector<PixelType, Ndim> &v1) const
		{
			Vector<PixelType, Ndim> v2;
			for (unsigned int i = 0; i < Ndim; i++)
			{
				v2[i] = this->data_[i] * v1[i];
			}
			return v2;
		}

		/** Overload operator+=. Add scalar. */
		void operator+=(const PixelType &scalar)
		  {
			for (unsigned int i = 0; i < Ndim; i++)
			{
				this->data_[i] += scalar;
			}
		  }

		/** Overload operator+=. Add vector. */
		void operator+=(const Vector<PixelType, Ndim> &v2)
		  {
			for (unsigned int i = 0; i < Ndim; i++)
			{
				this->data_[i] += v2[i];
			}
		  }

		/** Overload operator-=. Subtract scalar. */
		void operator-=(const PixelType &scalar)
		  {
			for (unsigned int i = 0; i < Ndim; i++)
			{
				this->data_[i] -= scalar;
			}
		  }

		/** Overload operator-=. Subtract vector. */
		void operator-=(const Vector<PixelType, Ndim> &v2)
		  {
			for (unsigned int i = 0; i < Ndim; i++)
			{
				this->data_[i] -= v2[i];
			}
		  }

		/** Overload operator*=. Multiplication with scalar. */
		void operator*=(const PixelType &scalar)
		  {
			for (unsigned int i = 0; i < Ndim; i++)
			{
				this->data_[i] *= scalar;
			}
		  }

		/** Overload operator*=. Multiplication with vector. */
		void operator*=(const Vector<PixelType, Ndim> &v2)
		  {
			for (unsigned int i = 0; i < Ndim; i++)
			{
				this->data_[i] *= v2[i];
			}
		  }

		/** Overload operator/. Division by vector. */
		Vector operator/(const PixelType scalar) const
		{
			IU_ASSERT(scalar != 0);
			PixelType invFactor = 1 / scalar;
			return operator*(invFactor);
		}

		/** Public copy constructor. */
		Vector(const Vector& from)
		{
			for (unsigned int i = 0; i < Ndim; i++)
				this->data_[i] = from[i];
		}

		/** Public copy assignment operator. */
		Vector& operator=(const Vector& from)
		{
			for (unsigned int i = 0; i < Ndim; i++)
				this->data_[i] = from[i];
			return *this;
		}
	};


	//namespace depricated{
	//	////////////////////////////////////////////////////////////////////////////////
	//	/** \brief Base class for N-dimensional unsigned int vectors (size vectors). */
	//	template<unsigned int Ndim>
	//	class SizeBase: public VectorBase<unsigned int, Ndim>
	//	{
	//	public:
	//		// Constructor. Init the size vector with ones.
	//		SizeBase() :
	//			VectorBase<unsigned int, Ndim>()
	//			{
	//			this->fill(1);
	//			}
	//		//
	//		  /** Special Constructor.
	//		   *  Init all elements of the size vector with a special value.
	//		   *  @param value value to initialize size vector elements.*/
	//		//  SizeBase(unsigned int value) :
	//		//      VectorBase<unsigned int, Ndim>(value)
	//		//  {
	//		//  }
	//		//
	//		//  /** Special Constructor.
	//		//   *  Init all elements of the vector with a initializer list.
	//		//   *  @param list Initializer list, e.g. {1,2,3}.*/
	//		//  SizeBase(std::initializer_list<unsigned int> list) :
	//		//      VectorBase<unsigned int, Ndim>(list)
	//		//  {
	//		//  }

	//		//using VectorBase<unsigned int, Ndim>::VectorBase;

	//		template <typename A0, typename... Args>
	//		SizeBase(A0 && a0, Args&&... args) : VectorBase<unsigned int, Ndim>(std::forward<A0>(a0), std::forward<Args>(args)...){
	//		}

	//		/** Destructor. */
	//		virtual ~SizeBase()
	//		{
	//		}

	//		/** Multiply all entries of the size vector to get the total number of elements. */
	//		unsigned int numel() const
	//		{
	//			unsigned int num_elements = this->data_[0];
	//			for (unsigned int i = 1; i < Ndim; i++)
	//			{
	//				if (this->data_[i] == 0)
	//				{
	//					std::stringstream msg;
	//					msg << "Zero size elements are not allowed. (" << *this << ")";
	//					throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
	//				}

	//				num_elements *= this->data_[i];
	//			}
	//			return num_elements;
	//		}

	//		/** Overload operator*. Multiplication with scalar and perform round operation. */
	//		template<typename ScalarType>
	//		SizeBase operator*(const ScalarType& scalar) const
	//		{
	//			SizeBase<Ndim> v;
	//			for (unsigned int i = 0; i < Ndim; i++)
	//			{
	//				v[i] = static_cast<unsigned int>(this->data_[i] * scalar + 0.5f);
	//			}
	//			return v;
	//		}

	//		/** Overload operator*=. Multiplication with scalar and perform round operation. */
	//		template<typename ScalarType>
	//		void operator*=(const ScalarType &scalar)
	//		{
	//			for (unsigned int i = 0; i < Ndim; i++)
	//			{
	//				this->data_[i] =
	//						static_cast<unsigned int>(this->data_[i] * scalar + 0.5f);
	//				;
	//			}
	//		}

	//		/** Overload operator/. Divide by scalar and perform round operation. */
	//		template<typename ScalarType>
	//		SizeBase operator/(const ScalarType& scalar) const
	//		{
	//			IU_ASSERT(scalar != 0);
	//			double invFactor = 1.0 / static_cast<double>(scalar);
	//			return operator*(invFactor);
	//		}

	//		/** Overload operator/=. Divide by scalar and perform round operation. */
	//		template<typename ScalarType>
	//		void operator/=(const ScalarType &scalar)
	//		{
	//			IU_ASSERT(scalar != 0);
	//			double invFactor = 1.0 / static_cast<double>(scalar);
	//			operator*=(invFactor);
	//		}

	//		/** Public copy constructor. */
	//		SizeBase(const SizeBase& from)
	//		{
	//			for (unsigned int i = 0; i < Ndim; i++)
	//				this->data_[i] = from[i];
	//		}

	//		/** Public copy assignment operator. */
	//		SizeBase& operator=(const SizeBase& from)
	//		{
	//			for (unsigned int i = 0; i < Ndim; i++)
	//				this->data_[i] = from[i];
	//			return *this;
	//		}

	//		/** Get data buffer. This is used in LinearDeviceMemory::KernelData,
	//		 *  ImageGpu::KernelData and VolumeGpu::KernelData to copy the size vector
	//		 *  from host to device.
	//		 */
	//		unsigned int* ptr()
	//		{
	//			return this->data_;
	//		}

	//		/** Get const data buffer. This is used in LinearDeviceMemory::KernelData,
	//		 *  ImageGpu::KernelData and VolumeGpu::KernelData to copy the size vector
	//		 *  from host to device.
	//		 */
	//		const unsigned int* ptr() const
	//		{
	//			return reinterpret_cast<const unsigned int*>(this->data_);
	//		}
	//	};

	//	////////////////////////////////////////////////////////////////////////////////
	//	/** \brief Main class for N-dimensional unsigned int vectors (size vectors). */
	//	template<unsigned int Ndim>
	//	class Size: public SizeBase<Ndim>
	//	{
	//	public:
	//		/** Constructor. */
	//		Size() = default;
	//		using SizeBase<Ndim>::SizeBase; // inherit all constructors of SizeBase

	//		//  Size() :
	//		//      SizeBase<Ndim>()
	//		//  {
	//		//  }
	//		//
	//		//  /** Special Constructor.
	//		//   *  Init all elements of the size vector with a special value.
	//		//   *  @param value value to initialize size vector elements.*/
	//		//  Size(unsigned int value) :
	//		//      SizeBase<Ndim>(value)
	//		//  {
	//		//  }
	//		//
	//		//  /** Special Constructor.
	//		//   *  Init all elements of the vector with a initializer list.
	//		//   *  @param list Initializer list, e.g. {1,2,3}.*/
	//		//  Size(std::initializer_list<unsigned int> list) :
	//		//      SizeBase<Ndim>(list)
	//		//  {
	//		//  }


	//		/** Destructor. */
	//		~Size()
	//		{
	//		}

	//		/** Public copy constructor. */
	//		Size(const Size& from) :
	//			SizeBase<Ndim>(from)
	//			{
	//			}

	//		/** Public copy constructor. */
	//		Size(const SizeBase<Ndim>& from) :
	//			SizeBase<Ndim>(from)
	//			{
	//			}

	//		/** Public copy assignment operator. */
	//		Size& operator=(const Size& from)
	//		{
	//			SizeBase<Ndim>::operator=(from);
	//			return *this;
	//		}

	//		/** Public copy assignment operator. */
	//		Size& operator=(const SizeBase<Ndim>& from)
	//		{
	//			SizeBase<Ndim>::operator=(from);
	//			return *this;
	//		}

	//	};

	//	////////////////////////////////////////////////////////////////////////////////
	//	/** \brief Template specialization for 2-d unsigned int vectors (size vectors).
	//	 *
	//	 * This class additionally has public members width, height and is used for
	//	 * the Image class to be compatible with previously written code.
	//	 */
	//	template<>
	//	class Size<2> : public SizeBase<2>
	//	{
	//	public:
	//		/** Width: Reference to 0th entry of data buffer */
	//		unsigned int& width;
	//		/** Height: Reference to 1st entry of data buffer */
	//		unsigned int& height;

	//		/** Constructor. */
	//		Size() :
	//			SizeBase<2>(), width(this->data_[0]), height(this->data_[1])
	//			{
	//			}

	//		/** Special Constructor.
	//		 *  Init all elements of the size vector with a special value.
	//		 *  @param value value to initialize size vector elements.*/
	//		Size(unsigned int value) :
	//			SizeBase<2>(value), width(this->data_[0]), height(this->data_[1])
	//			{
	//			}

	//		using SizeBase<2>::SizeBase;

	//		/** Special Constructor.
	//		 *  Init all elements of the vector with a initializer list.
	//		 *  @param list Initializer list, e.g. {1,2}.*/
	//		Size(std::initializer_list<unsigned int> list) :
	//			SizeBase<2>(list), width(this->data_[0]), height(this->data_[1])
	//			{
	//			}

	//		/** Special Constructor. Init size with width, height.
	//		 *  @param width Set 0th entry of data buffer
	//		 *  @param height Set 1st entry of data buffer
	//		 */
	//		Size(unsigned int width, unsigned int height) :
	//			SizeBase<2>(), width(this->data_[0]), height(this->data_[1])
	//			{
	//			data_[0] = width;
	//			data_[1] = height;
	//			}

	//		template<class other, class = typename std::enable_if<std::is_class<other>::value>::type>
	//		explicit Size(const other & x): width(this->data_[0]), height(this->data_[1]){
	//			for(unsigned int i=0; i< 2; ++i){
	//				(*this)[i] = x[i];
	//			};
	//		}

	//		/** Destructor. */
	//		~Size()
	//		{
	//		}

	//		/** Public copy constructor. */
	//		Size(const Size& from) :
	//			SizeBase<2>(from), width(this->data_[0]), height(this->data_[1])
	//			{
	//			}

	//		/** Public copy constructor. */
	//		Size(const SizeBase& from) :
	//			SizeBase<2>(from), width(this->data_[0]), height(this->data_[1])
	//			{
	//			}

	//		/** Public copy assignment operator. */
	//		Size& operator=(const Size& from)
	//		{
	//			SizeBase::operator=(from);
	//			return *this;
	//		}

	//		/** Public copy assignment operator. */
	//		Size& operator=(const SizeBase<2>& from)
	//		{
	//			SizeBase::operator=(from);
	//			return *this;
	//		}
	//	};

	//	////////////////////////////////////////////////////////////////////////////////
	//	/** \brief Template specialization for 3-d unsigned int vectors (size vectors).
	//	 *
	//	 * This class additionally has public members width, height, depth  and is used for
	//	 * the Volume class to be compatible with previously written code.
	//	 */
	//	template<>
	//	class Size<3> : public SizeBase<3>
	//	{
	//	public:
	//		/** Width: Reference to 0th entry of data buffer */
	//		unsigned int& width;
	//		/** Height: Reference to 1st entry of data buffer */
	//		unsigned int& height;
	//		/** Depth: Reference to 2nd entry of data buffer */
	//		unsigned int& depth;


	//		/** Constructor. */
	//		Size() :
	//			SizeBase<3>(), width(this->data_[0]), height(this->data_[1]),
	//			depth(this->data_[2])
	//			{
	//			}

	//		/** Special Constructor.
	//		 *  Init all elements of the size vector with a special value.
	//		 *  @param value value to initialize size vector elements.*/
	//		Size(unsigned int value) :
	//			SizeBase<3>(value), width(this->data_[0]), height(this->data_[1]),
	//			depth(this->data_[2])
	//			{
	//			}

	//		/** Special Constructor.
	//		 *  Init all elements of the vector with a initializer list.
	//		 *  @param list Initializer list, e.g. {1,2,3}.*/
	//		Size(std::initializer_list<unsigned int> list) :
	//			SizeBase<3>(list), width(this->data_[0]), height(this->data_[1]),
	//			depth(this->data_[2])
	//			{
	//			}

	//		/** Special Constructor. Init size with width, height, depth.
	//		 *  @param width Set 0th entry of data buffer
	//		 *  @param height Set 1st entry of data buffer
	//		 *  @param depth Set 2nd entry of data buffer
	//		 */
	//		Size(unsigned int width, unsigned int height, unsigned int depth) :
	//			SizeBase<3>(), width(this->data_[0]), height(this->data_[1]),
	//			depth(this->data_[2])
	//			{
	//			data_[0] = width;
	//			data_[1] = height;
	//			data_[2] = depth;
	//			}

	//		template<class other, class = typename std::enable_if<std::is_class<other>::value>::type>
	//		explicit Size(const other & x): width(this->data_[0]), height(this->data_[1]), depth(this->data_[2]){
	//			for(unsigned int i=0; i< 3; ++i){
	//				(*this)[i] = x[i];
	//			};
	//		}


	//		/** Destructor. */
	//		~Size()
	//		{
	//		}

	//		/** Public copy constructor. */
	//		Size(const Size& from) :
	//			SizeBase<3>(from), width(this->data_[0]), height(this->data_[1]),
	//			depth(this->data_[2])
	//			{
	//			}

	//		/** Public copy constructor. */
	//		Size(const SizeBase& from) :
	//			SizeBase<3>(from), width(this->data_[0]), height(this->data_[1]),
	//			depth(this->data_[2])
	//			{
	//			}

	//		/** Public copy assignment operator. */
	//		Size& operator=(const Size& from)
	//		{
	//			SizeBase::operator=(from);
	//			return *this;
	//		}

	//		/** Public copy assignment operator. */
	//		Size& operator=(const SizeBase<3>& from)
	//		{
	//			SizeBase::operator=(from);
	//			return *this;
	//		}
	//	};
	//}; //depricated

}  //namespace iu

