#ifndef CONFIG_H
#define CONFIG_H

#cmakedefine DATA_DIR "@DATA_DIR@"

#define DATA_PATH(file) DATA_DIR "/" file

#cmakedefine RESULTS_DIR "@RESULTS_DIR@"

#define RESULTS_PATH(file) RESULTS_DIR "/" file

#undef BOOST_NO_INCLASS_MEMBER_INITIALIZATION

#endif // CONFIG_H
