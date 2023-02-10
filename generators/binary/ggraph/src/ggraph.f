      program ggraph
c     ---------------------------------------------------------------
c     generator: max-flow min-cost directed grid graph
c
c     Author: Mauricio G.C. Resende
c             AT&T Bell Laboratories
c             mgcr@research.att.com
c
c     Date: 23 Sep 91
c           01 Feb 92 - modified to generate COST in [1,MAXCOST]
c                                            CAP in [1,MAXCAP]
c 
c     standard input: h, w, MAXCAP, MAXCOST, SEED
c
c     graph has h * w + 2 nodes: source s, sink t and a grid of
c     h by w nodes.
c
c     arcs go from s to h nodes of first grid column
c             from h nodes of last grid column to t
c             from (i,j) to (i+1,j) and (j+1,i) except
c             for last row that go from (i,h) to (i+1,h) and last
c             column that go from (w,j) to t
c
c     arc capacities for grid arcs uniformly distributed [0,MAXCAP]
c                    for s - (1,j) = INFINITY
c                    for (w,j) - t = INFINITY
c
c     arc costs for grid arcs uniformly distributed [0,MAXCOST]
c               for s - (1,j) = 0
c               for (w,j) - t = 0
c     
c     Supply at node s = MAXFLO(s-t)
c     Demand at node t = MAXFLO(s-t)
c     Supply and demands for all other nodes = 0
c
c     Output: DIMACS min format
c
c     Uses subroutines: dnfwd and dnsub to compute max-flow.
c                       Reference: D. Goldbarb, M.D. Grigoriadis,
c                       "A computational comparison of the dinic 
c                       and network simplex methods for maximum flow",
c                       Annals of Operations Research 7, 1988.
c                       
c                       rand - portable pseudo-random number generator.
c                       Reference: L. Schrage, "A More Portable FORTRAN
c                       Random Number Generator", ACM Transactions on
c                       Mathematical Software, June, 1979.
c     
c     ---------------------------------------------------------------

      integer maxh,maxw,maxarc
      parameter (maxh=1000,maxw=1000)
      parameter (maxarc=524273)
      integer caps(maxh),costs(maxh),capt(maxh),costt(maxh)
      integer h,w,maxf,maxc,n1,n2,cap,unif,i,j
      integer nnodes,narcs,snode,tnode,seed,seed0
      integer cost(524273)
      integer v10(maxarc),v20(maxarc),cap0(maxarc),cost0(maxarc)
C
C************START DNSUB STANDARD USER XFACE DECLARATION BLOCK*********
C
C      USER MUST INCLUDE THIS BLOCK INHIS/HER	CALLING	PROGRAM.
C
C   THESE DECLARATIONS ACCOUNT FOR THE ENTIRE ARRAY STORAGE USED BY
C   THE      DNSUB SUBROUTINES, INCLUDING WHAT IS NECESSARY TO STORETHE
C   INPUT DATA.      IT AMOUNTS TO 6*NODES +5*ARCS WORDS, BUT IT MAY BE
C   STATED MORE      PRECISELY AS FOLLOWS:
C
C   1) FOR SOLVING PROBLEMS WITH UP TO 32,765 NODES AND      2,147,483,647
C   ARCS, AND WITH FLOW      VALUES OF UP TO2,147,483,647, IT SUFFICES
C   TO DECLARE ARRAYS AS SHOWN UNDER 'STD ARRANGEMENT' IN THE
C   TABLE BELOW.  THIS ARRANGEMENT IS THE ONE USED IN THE RELEASE
C   VERSION OF THE DNSUB SUBROUTINES.
C
C   2) IN ORDER      TO HANDLE PROBLEMS WITHMORE THAN 32,765 NODES,	ALL
C   INTEGER*2 DECLARATIONS BELOW MUST BE CHANGED TO INTEGER*4 AND
C   THE      DNSUB SUBROUTINES AND USER CALLING PROGRAMS MUST BE COMPILED
C   AND      LINKED.(SEE ALTERNATE 1 BELOW.)
C
C   3) FOR SOLVING ONLY      PROBLEMS WITH AT MOST 32,765 NODES, 32,767
C   ARCS, AND WITH FLOWS NOT EXCEEDING 32,767, ALL INTEGER*4
C   DECLARATIONS BELOW MAY BE CHANGED TO INTEGER*2, AND      THE DNSUB
C   SUBROUTINES       AND USER CALLING PROGRAMS MUSTBE COMPILED AND
C   LINKED. (SEE ALTERNATE 2 BELOW.)  IT IS ALSO NECESSARY TO
C   ACTIVATE THE STATEMENT DNIBIG=32767      in subroutine dnsub, and to
C   include 'implicit INTEGER*2 (i-n)' statements in each program
C   unit.  THIS      ARRANGEMENT, WHICH USESTHE LEAST AMOUNT OF MEMORY
C   AND      IS THE FASTEST when executed on16-BIT PROCESSORS, IS ONLY
C   USEFUL FOR A SET OF      VERY SPECIAL APPLICATIONS.  ITSgeneral
C   use      is not recommended.  OTHER ALTERNATES ARE ALSO POSSIBLE.
C
C  -----------------------------------------------------------------
C
C        STD ARRANGEMENT    ALTERNATE 1      ALTERNATE	2
C      				   (not	recommended)
C        ---------------  ---------------  ----------------
C  ARRAY   LENGTH TYPE EQV'D TYPE  TYPE EQV'D TYPE  TYPE EQV'D TYPE
C  ------  -----    ----  -------  ---- ----- ----  ---- ----- -----
C  DNFAPT    N      4               4                2
C  DNFADJ    A      2               4                2
C  DNCAP     A      4               4                2
C  DNBAPT    N      4               4                2
C  DNBADJ    A      2 DNFROM  2     4 DNFROM  4      2 DNFROM   2
C  DNFLOW    A      4 DNTO    2     4 DNTO    4      2 DNTO     2
C  DNBTOF    A      4 DNGCAP  4     4 DNGCAP  4      2 DNGCAP   2
C  DNPTRF    N      4               4                2
C  DNPTRB    N      4               4                2
C  DNLIST    N      2               4                2
C  DNFLAB    N      4 DNDIST  2     4 DNDIST  4      2 DNDIST   2
C  -----------------------------------------------------------------
C
C  TOTAL BYTES:     22*N + 16*A     24*N + 20*A       12*N + 10*A
C  (N=NODES, A=ARCS)
C  -----------------------------------------------------------------
C
C DIMENSION THESE TO AT LEAST MAX NUMBER OF NODES + 2:
        INTEGER   DNLIST(65538),DNDIST(65538)
        INTEGER   DNFAPT(65538),DNPTRF(65538),DNBAPT(65538),
     *              DNFLAB(65538),DNPTRB(65538)
C DIMENSION THESE TO AT LEAST MAX NUMBER OF ARCS + 1:
      INTEGER DNFADJ(524273),DNFROM(524273),DNTO(524273),DNBADJ(524273)
      INTEGER DNCAP(524273),DNGCAP(524273),DNFLOW(524273),DNBTOF(524273)
C
        COMMON /DN01/DNFAPT/DN02/DNFADJ/DN03/DNCAP/DN04/DNFROM
        COMMON /DN05/DNFLOW/DN06/DNGCAP/DN08/DNPTRF/DN09/DNLIST
        COMMON /DN10/DNBAPT/DN11/DNFLAB/DN12/DNPTRB
C
        INTEGER  DNNODE,NODP12,DNSRCE,DNSINK,I2,II2,K2,J2,IHEAD2,
     *  ITAIL2,QHEAD2,QTAIL2,KP2,KSAT2,DNNOP2,LAYMAX,K2DST,K2DP1,MAXD2
        INTEGER  DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,DNIBIG
        COMMON /DN00/DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,
     *               DNELTM,DNIBIG,DNNODE,DNSRCE,DNSINK,DNNOP2
        EQUIVALENCE (DNFLOW(1),DNTO(1)),(DNFLAB(1),DNDIST(1)),
     *              (DNFROM(1),DNBADJ(1)),(DNGCAP(1),DNBTOF(1))
C
C************END DNSUB STANDARD USER XFACE DECLARATION BLOCK***********
 
      read (5,*) h,w,maxf,maxc,seed
      seed0=seed
      nnodes=h*w+2
      narcs=2*(h-1)*w+w+h
      snode=h*w+1
      tnode=h*w+2
c     for dinic interface
c     ------------------------------------------------------
      nodes = nnodes
      isou=snode
      isnk=tnode
c     ------------------------------------------------------

      numarc=0
      do 1010 i=1,h-1
           numarc=numarc+1
           dnfrom(numarc)=(i-1)*w+1
           dnto(numarc)=dnfrom(numarc)+1
           dngcap(numarc)=unif(maxf,seed)
           cost(numarc)=unif(maxc,seed)
           caps(i)=dngcap(numarc)
           numarc=numarc+1
           dnfrom(numarc)=(i-1)*w+1
           dnto(numarc)=dnfrom(numarc)+w
           dngcap(numarc)=unif(maxf,seed)
           cost(numarc)=unif(maxc,seed)
           caps(i)=caps(i)+dngcap(numarc)
           costs(i)=unif(maxc,seed)
1010  continue
      numarc=numarc+1
      dnfrom(numarc)=(h-1)*w+1
      dnto(numarc)=dnfrom(numarc)+1
      dngcap(numarc)=unif(maxf,seed)
      cost(numarc)=unif(maxc,seed)
      caps(h)=dngcap(numarc)
      costs(h)=unif(maxc,seed)
      do 1020 i=1,h-1
           do 1030 j=2,w-2
                numarc=numarc+1
                dnfrom(numarc)=(i-1)*w+j
                dnto(numarc)=dnfrom(numarc)+1
                dngcap(numarc)=unif(maxf,seed)
                cost(numarc)=unif(maxc,seed)
                numarc=numarc+1
                dnfrom(numarc)=(i-1)*w+j
                dnto(numarc)=dnfrom(numarc)+w
                dngcap(numarc)=unif(maxf,seed)
                cost(numarc)=unif(maxc,seed)
1030       continue
1020  continue
      do 1040 j=2,w-2
           numarc=numarc+1
           dnfrom(numarc)=(h-1)*w+j
           dnto(numarc)=dnfrom(numarc)+1
           dngcap(numarc)=unif(maxf,seed)
           cost(numarc)=unif(maxc,seed)
1040  continue
        
           
      do 1050 i=1,h-1
           numarc=numarc+1
           dnfrom(numarc)=(i-1)*w+w-1
           dnto(numarc)=dnfrom(numarc)+1
           dngcap(numarc)=unif(maxf,seed)
           cost(numarc)=unif(maxc,seed)
           capt(i)=dngcap(numarc)
           numarc=numarc+1
           dnfrom(numarc)=(i-1)*w+w-1
           dnto(numarc)=dnfrom(numarc)+w
           dngcap(numarc)=unif(maxf,seed)
           cost(numarc)=unif(maxc,seed)
           costt(i)=unif(maxc,seed)
1050  continue
      numarc=numarc+1
      dnfrom(numarc)=(h-1)*w+w-1
      dnto(numarc)=dnfrom(numarc)+1
      dngcap(numarc)=unif(maxf,seed)
      cost(numarc)=unif(maxc,seed)
      capt(h)=dngcap(numarc)
      costt(h)=unif(maxc,seed)


      do 1060 i=1,h-1
           numarc=numarc+1
           dnfrom(numarc)=(i-1)*w+w
           dnto(numarc)=dnfrom(numarc)+w
           dngcap(numarc)=unif(maxf,seed)
           cost(numarc)=unif(maxc,seed)
           capt(i+1)=capt(i+1)+dngcap(numarc)
1060  continue

      do 1070 i=1,h
           numarc=numarc+1
           dnfrom(numarc)=h*w+1
           dnto(numarc)=(i-1)*w+1
           dngcap(numarc)=caps(i)
           cost(numarc)=costs(i)
1070  continue

      do 1080 i=1,h
           numarc=numarc+1
           dnfrom(numarc)=i*w
           dnto(numarc)=h*w+2
           dngcap(numarc)=capt(i)
           cost(numarc)=costt(i)
1080  continue

      do 90101 i=1,numarc
         v10(i)=dnfrom(i)
         v20(i)=dnto(i)
         cap0(i)=dngcap(i)
         cost0(i)=cost(i)
90101  continue
      call dnfwd(nodes,narcs,iretn)
      call dnsub(nodes,narcs,isou,isnk,maxflo,numaug,nphase,
     +           nncut,nacut,etime0,iretn)
      write(6,100) h,w
100   format("c    max-flow min-cost on a",i3," by ",i3," grid")
      write(6,105) maxf,maxc,seed0
105   format("c    cap: [0,",i8,"]   cost: [0,",i8,"]   seed:", i10)
      write(6,101) nodes,narcs
101   format("p min ",2i15)
      write(6,103) isou,maxflo
103   format("n ",2i15)
      write(6,103) isnk,-maxflo
      do 9010 i=1,numarc
         write(6,102) v10(i),v20(i),cap0(i),cost0(i)
102      format("a ",2i10," 0 ",2i10)
9010  continue
      
      stop
      end

      integer function unif(max,seed)
      integer max,seed
      real rand
      unif=1+rand(seed)*(max-1)
      return
      end

       real function rand(ix)
c      =================================================================
c      Portable pseudo-random number generator.
c      Reference: l. schrage, "A More Portable FORTRAN
c      Random Number Generator", ACM Transactions on
c      Mathematical Software, June, 1979.
c      =================================================================

       integer*4 a,p,ix,b15,b16,xhi,xalo,leftlo,fhi,k
       data a/16807/,b15/32768/,b16/65536/,p/2147483647/

       xhi=ix/b16
       xalo=(ix-xhi*b16)*a
       leftlo=xalo/b16
       fhi=xhi*a+leftlo
       k=fhi/b15
       ix=(((xalo-leftlo*b16)-p)+(fhi-k*b15)*b16)+k
       if (ix.lt.0) ix=ix+p

       rand=float(ix)*4.656612875e-10

       return
       end

      SUBROUTINE DNSUB(NODES,NARCS,ISOURC,ISINK,MAXFLO,
     *         NUMAUG,NUMSTG,NNCUT,NACUT,ELTIM,IRETN)
C
C************START DNSUB STANDARD USER XFACE DECLARATION BLOCK*********
C
C      USER MUST INCLUDE THIS BLOCK INHIS/HER	CALLING	PROGRAM.
C
C   THESE DECLARATIONS ACCOUNT FOR THE ENTIRE ARRAY STORAGE USED BY
C   THE      DNSUB SUBROUTINES, INCLUDING WHAT IS NECESSARY TO STORETHE
C   INPUT DATA.      IT AMOUNTS TO 6*NODES +5*ARCS WORDS, BUT IT MAY BE
C   STATED MORE      PRECISELY AS FOLLOWS:
C
C   1) FOR SOLVING PROBLEMS WITH UP TO 32,765 NODES AND      2,147,483,647
C   ARCS, AND WITH FLOW      VALUES OF UP TO2,147,483,647, IT SUFFICES
C   TO DECLARE ARRAYS AS SHOWN UNDER 'STD ARRANGEMENT' IN THE
C   TABLE BELOW.  THIS ARRANGEMENT IS THE ONE USED IN THE RELEASE
C   VERSION OF THE DNSUB SUBROUTINES.
C
C   2) IN ORDER      TO HANDLE PROBLEMS WITHMORE THAN 32,765 NODES,	ALL
C   INTEGER*2 DECLARATIONS BELOW MUST BE CHANGED TO INTEGER*4 AND
C   THE      DNSUB SUBROUTINES AND USER CALLING PROGRAMS MUST BE COMPILED
C   AND      LINKED.(SEE ALTERNATE 1 BELOW.)
C
C   3) FOR SOLVING ONLY      PROBLEMS WITH AT MOST 32,765 NODES, 32,767
C   ARCS, AND WITH FLOWS NOT EXCEEDING 32,767, ALL INTEGER*4
C   DECLARATIONS BELOW MAY BE CHANGED TO INTEGER*2, AND      THE DNSUB
C   SUBROUTINES       AND USER CALLING PROGRAMS MUSTBE COMPILED AND
C   LINKED. (SEE ALTERNATE 2 BELOW.)  IT IS ALSO NECESSARY TO
C   ACTIVATE THE STATEMENT DNIBIG=32767      in subroutine dnsub, and to
C   include 'implicit INTEGER*2 (i-n)' statements in each program
C   unit.  THIS      ARRANGEMENT, WHICH USESTHE LEAST AMOUNT OF MEMORY
C   AND      IS THE FASTEST when executed on16-BIT PROCESSORS, IS ONLY
C   USEFUL FOR A SET OF      VERY SPECIAL APPLICATIONS.  ITSgeneral
C   use      is not recommended.  OTHER ALTERNATES ARE ALSO POSSIBLE.
C
C  -----------------------------------------------------------------
C
C        STD ARRANGEMENT    ALTERNATE 1      ALTERNATE	2
C      				   (not	recommended)
C        ---------------  ---------------  ----------------
C  ARRAY   LENGTH TYPE EQV'D TYPE  TYPE EQV'D TYPE  TYPE EQV'D TYPE
C  ------  -----    ----  -------  ---- ----- ----  ---- ----- -----
C  DNFAPT    N      4               4                2
C  DNFADJ    A      2               4                2
C  DNCAP     A      4               4                2
C  DNBAPT    N      4               4                2
C  DNBADJ    A      2 DNFROM  2     4 DNFROM  4      2 DNFROM   2
C  DNFLOW    A      4 DNTO    2     4 DNTO    4      2 DNTO     2
C  DNBTOF    A      4 DNGCAP  4     4 DNGCAP  4      2 DNGCAP   2
C  DNPTRF    N      4               4                2
C  DNPTRB    N      4               4                2
C  DNLIST    N      2               4                2
C  DNFLAB    N      4 DNDIST  2     4 DNDIST  4      2 DNDIST   2
C  -----------------------------------------------------------------
C
C  TOTAL BYTES:     22*N + 16*A     24*N + 20*A       12*N + 10*A
C  (N=NODES, A=ARCS)
C  -----------------------------------------------------------------
C
C DIMENSION THESE TO AT LEAST MAX NUMBER OF NODES + 2:
        INTEGER   DNLIST(65538),DNDIST(65538)
        INTEGER   DNFAPT(65538),DNPTRF(65538),DNBAPT(65538),
     *              DNFLAB(65538),DNPTRB(65538)
C DIMENSION THESE TO AT LEAST MAX NUMBER OF ARCS + 1:
      INTEGER DNFADJ(524273),DNFROM(524273),DNTO(524273),DNBADJ(524273)
      INTEGER DNCAP(524273),DNGCAP(524273),DNFLOW(524273),DNBTOF(524273)
C
        COMMON /DN01/DNFAPT/DN02/DNFADJ/DN03/DNCAP/DN04/DNFROM
        COMMON /DN05/DNFLOW/DN06/DNGCAP/DN08/DNPTRF/DN09/DNLIST
        COMMON /DN10/DNBAPT/DN11/DNFLAB/DN12/DNPTRB
C
        INTEGER  DNNODE,NODP12,DNSRCE,DNSINK,I2,II2,K2,J2,IHEAD2,
     *  ITAIL2,QHEAD2,QTAIL2,KP2,KSAT2,DNNOP2,LAYMAX,K2DST,K2DP1,MAXD2
        INTEGER  DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,DNIBIG
        COMMON /DN00/DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,
     *               DNELTM,DNIBIG,DNNODE,DNSRCE,DNSINK,DNNOP2
        EQUIVALENCE (DNFLOW(1),DNTO(1)),(DNFLAB(1),DNDIST(1)),
     *              (DNFROM(1),DNBADJ(1)),(DNGCAP(1),DNBTOF(1))
C
C************END DNSUB STANDARD USER XFACE DECLARATION BLOCK***********
C***********************************************************************
C
C CALLING CONDITIONS:
C       USER CALLABLE WITH INPUT DATA AS FOLLOWS.
C INPUT:
C   SCALARS (IN CALLING SEQUENCE; ALL INTEGER*4):
C       NODES:  NUMBER OF NODES (INCLUDING SOURCE AND SINK)
C       NARCS:  NUMBER OF ARCS
C       ISOURC: NODE NUMBER FOR SOURCE
C       ISINK:  NODE NUMBER FOR SINK
C
C   ARRAYS (IN COMMON):
C       DNFAPT: (NODES+1)-LONG INTEGER*4 POINTER ARRAY FOR FORWARD ADJA-
C               CENCY LISTS (I.E. THE FORWARD ADJACENCY LIST OF A NODE I
C               IS THE SET OF ARCS
C                      {(I,DNFADJ(J)) :J=DNFAPT(I),...,DNFAPT(I+1)-1 }
C               NOTE: MUST HAVE DNFAPT(NODES+1) = NARCS+1.
C
C       DNFADJ: NARCS-LONG INTEGER*2 ARRAY GIVING THE LIST OF NODES IN
C               THE FORWARD ADJACENCY LIST DNFADJ(J) FOR EACH NODE J,
C               AS DESCRIBED ABOVE.
C
C       DNCAP:  NARCS-LONG INTEGER*4 ARRAY GIVING THE ARC CAPACITIES, IN
C               THE ORDER PRESCRIBED BY DNFAPT(.) AND DNFADJ(.). ALL ARC
C               CAPACITIES MUST BE GIVEN AS POSITIVE INTEGERS.
C
C   NOTE:       FOR TRANSFORMING UNORDERED ARC LISTS TO THE ABOVE INPUT
C               DATA STRUCTURE,USE SUBROUTINE DNFWD BEFORE CALLING DNSUB
C
C OUTPUT:
C
C   SCALARS (IN CALLING SEQUENCE):
C       MAXFLO: VALUE OF MAXIMUM FLOW (INTEGER*4).
C       NUMAUG: NUMBER OF FLOW AUGMENTATIONS (INTEGER*4).
C       NUMSTG: NUMBER OF STAGES (LAYERED NETWORKS CREATED; INTEGER*4).
C       NNCUT:  NUMBER OF NODES ON source SIDE OF FINAL CUT (INTEGER*4)
C       NACUT:  NUMBER OF SATURATED ARCS IN THE FINAL CUT (INTEGER*4)
C       ELTIM:  EXECUTION TIME, IN SECONDS (SEE NOTE 3 BELOW) (REAL*4).
C       IRETN:  NONZERO IF THERE ARE ERRORS IN INPUT DATA.
C
C   ARRAYS (IN COMMON:):
C       DNFAPT: THE ORIGINAL DNFAPT(.) WITH SOME OF ITS ELEMENTS
C               NEGATED TO MARK THE NODES THAT ARE ON THE SINK SIDE OF
C               THE MINIMUM CUT, I.E. DNFAPT(I)<0 IMPLIES THAT NODE I IS
C               ON THE SINK SIDE OF CUT, ELSE, IT IS ON THE source SIDE.
C
C       DNFADJ: THE ORIGINAL DNFADJ(.) WITH SOME OF ITS ELEMENTS NEGATED
C               TO MARK THOSE SATURATED ARCS THAT ARE IN THE MIN CUT
C               FOUND BY THE ALGORITHM, I.E.  DNFADJ(I)<0 IMPLIES THAT
C               THE I-TH ARC IN FORWARD ADJACENY ORDER IS IN THE CUT;
C               THE FOLLOWING CODE SEGMENT PRINTS THESE ARCS:
C
C                               DO 2 K=1,NODES
C                                  IBEG=IABS(DNFAPT(K))
C                                  IEND=IABS(DNFAPT(K+1))-1
C                                  DO 1 I=IBEG,IEND
C                                    IF(DNFADJ(I).LT.0)PRINT THE ARC
C                       1          CONTINUE
C                       2       CONTINUE
C
C               (SEE NOTE 1 BELOW).
C
C       DNCAP:  THE ORIGINAL CAPACITIES, UNALTERED.
C
C       DNFLOW: NARCS-LONG INTEGER*4 ARRAY WHICH GIVES FLOWS ON ARCS, IN
C               FORWARD ADJACENCY ORDER.
C
C       DNBAPT: (NODES+1)-LONG INTEGER*4 POINTER ARRAY FOR BACKWARD
C               ADJACENCY LISTS (I.E. THE FORWARD ADJACENCY LIST OF
C               NODE I IS THE SET OF ARCS
C                   {(DNBADJ(J),J): J=DNBAPT(I),...,DNBAPT(I+1)-1 }.
C               NOTE: WE MUST HAVE DNBAPT(NODES+1) = NARCS+1.
C
C       DNBADJ: NARCS-LONG INTEGER*2 ARRAY GIVING THE BACKWARD ADJACENCY
C               LIST OF EACH NODE J, ONE AFTER THE OTHER, AS DESCRIBED
C               ABOVE (ALSO SEE NOTE 2 BELOW).
C
C       DNBTOF: NARCS-LONG INTEGER*4 ARRAY GIVING THE BACKWARD ADJACENCY
C               TO FORWARD ADJACENCY MAPPING, I.E. THE J-TH ARC
C               IN THE BACKWARD ADJACENCY ORDER IS THE DNBTOF(J)-TH
C               ARC IN THE FORWARD ADJACENCY ORDER.
C
C***********************************************************************
C
C INTERNAL SCRATCH ARRAYS (IN COMMON):
C
C       DNLIST: (NODES+1)-LONG INTEGER*2 ARRAY USED AS FOLLOWS:
C               1) AS THE QUEUE IN BFS SEARCH FOR CONSTRUCTING THE
C                  LAYERED GRAPH (SUBR. DNBFS).
C               2) AS THE "PARENT" ARRAY IN DFS SEARCH OF LAYERED
C                  GRAPH (SUBR: DNDFS, DNPUSH).
C               3) AS SCRATCH IN CONSTRUCTING FWD ADJACENCIES
C                  (SUBR DNFWD), AND BWD ADJACENCIES (SUBR. DNSUB).
C
C       DNFLAB: NODES-LONG INTEGER*4 ARRAY USED IN DFS SEARCH OF
C               LAYERED GRAPH TO STORE THE FLOW LABEL FOR EACH NODE
C               (SEE ARRAY DNDIST() ).
C
C       DNDIST: NODES-LONG INTEGER*2 ARRAY USED IN CONSTRUCTING THE
C               LAYERED GRAPH BY BFS, STORING THE DISTANCE OF EACH
C               NODE FROM THE SOURCE; DNDIST(ISOURC)=0 (EQUIVALENCED
C               TO THE INTEGER*4 ARRAY DNFLAB() ).
C
C       DNPTRF: NODES-LONG INTEGER*4 STATUS AND POINTER ARRAY, I.E.
C               DNPTRF(K)=0 IF THERE ARE NO OPEN ARCS LEAVING NODE K
C                               IN THE REPRESENTATION OF THE LAYERED
C                               NETWORK.
C                        >0 THE ARC INDEX, IN 'FA' ORDER, THAT
C                               CORRESPONDS TO THE LAST ARC IN THE
C                               FWD ADJACENCY OF K WHICH IS SCANNED
C                               BACKWARD.
C
C       DNPTRB: NODES-LONG INTEGER*4 STATUS AND POINTER ARRAY, I.E.
C               DNPTRB(K)=0 IF THERE ARE NO OPEN ARCS ENTERING NODE K
C                               IN THE REPRESENTATION OF THE LAYERED
C                               NETWORK.
C                        >0 THE ARC INDEX (IN 'BA' ORDER) OF THE FIRST
C                               ARC IN THE BWD ADJACENCY OF K TO BE
C                               SCANNED IN LIFO ORDER.
C
C***********************************************************************
C
C NOTES:1) DNFADJ(J) IS INTERNALLY NEGATED TO MARK THE J-TH ARC (IN
C          FORWARD ADJACENCY ORDER) AS 'CLOSED'. BEFORE RETURNING
C          TO THE CALLING PGM, THESE MARKINGS ARE DISCARDED, AND SOME
C          ELEMENTS OF DNFADJ(.) ARE NEGATED TO REFLECT THE OUTPUT
C          SPECIFICATION DESCRIBED ABOVE.
C
C       2) DNBADJ(J) IS INTERNALLY NEGATED TO MARK THE J-TH ARC
C          (IN BACKWARD ADJACENCY ORDER) AS 'CLOSED'.
C
C       3) THE USER MUST PROVIDE A TIMING SUBROUTINE THAT RETURNS
C          THE CURRENT TIME IN VARIABLE ' T ', IN SECONDS.  SUCH
C          A SUBROUTINE IS INSTALLATION-DEPENDENT AND IS NOT GIVEN HERE.
C          IF THIS CANNOT BE DONE, PROVIDE THE FOLLOWING SUBROUTINE AND
C          FORGET THE EXECUTION TIMING INFORMATION:
C
C               SUBROUTINE GETIME(T)
C               REAL T
C               T=0.0
C               RETURN
C               END
C
C***********************************************************************
C
C       SUBROUTINES CALLED:     DNBFS, DNDFS, DNCUT, GETIME,DNCLEA
C
C***********************************************************************
C
C   NAMES COLLECTIVELY RESERVED BY ALL DNSUB SUBROUTINES:
C
C   DN00, DN01, DN02, DN03, DN04, DN05, DN06, DN08, DN09, DN10, DN11,
C   DN12, DNARC, DNAUG, DNBADJ, DNBAPT, DNBFS, DNBTOF, DNCAP, DNCLEA,
C   DNCUT, DNDFS, DNDIST, DNFADJ, DNPUSH, DNFAPT, DNFLAB, DNFLOW,
C   DNFROM, DNFVA, DNFWD, DNGCAP, DNIBIG, DNLAUG, DNLFVA, DNLIST,
C   DNNODE, DNNOP2, DNOUT, DNPTRB, DNPTRF, DNSINK, DNSRCE, DNSTGE,
C   DNSUB, DNELTM, DNTO.
C
C**********************************************************************
C INITIALIZATION:
C
C       CALL GETIME(TIMBEG)
C        TIMBEG= time(it)

        IRETN=0
        DNIBIG=2147483647
C       DNIBIG=32767
        DNARC=NARCS
        DNNODE=NODES
        DNNOP2=DNNODE+2
        DNSRCE=ISOURC
        DNSINK=ISINK
        DO 200 I4=1,DNARC
200     DNFLOW(I4)=0
C***********************************************************************
C
C USING THE FORWARD ADJACENCY ARRAYS DNFAPT(.) AND DNFADJ(.) (IN
C COMMON), CREATE THE BACKWARD ADJACENCY ARRAYS DNBAPT(.), DNBADJ(.)
C AND THE BACKWARD TO FORWARD ADJACENCY MAPPING DNBTOF(.).  DNLIST(.) IS
C USED HERE AS A SCRATCH ARRAY.
C
        NODP12=DNNODE+1
        DO 210 I2=1,NODP12
                DNLIST(I2)=0
210             DNBAPT(I2)=0
C TEMPORARILY STORE IN DNBAPT(.) NUMBER OF ARCS INTO EACH NODE:
        DO 230 I2=1,DNNODE
                IBEG4=DNFAPT(I2)
                IEND4=DNFAPT(I2+1)-1
                DO 220 I4=IBEG4,IEND4
                  JJDN=DNFADJ(I4)
220             DNBAPT(JJDN)=DNBAPT(JJDN)+1
230     CONTINUE
C CONSTRUCT DNBAPT(.):
        IHPI4=DNBAPT(1)
        DNBAPT(1)=1
        DO 240 I2=1,DNNODE
                IHPSV4=IHPI4+DNBAPT(I2)
                IHPI4=DNBAPT(I2+1)
240     DNBAPT(I2+1)=IHPSV4
C CONSTRUCT DNBADJ(.) AND DNBTOF(.):
        DO 260 I2=1,DNNODE
                IBEG4=DNFAPT(I2)
                IEND4=DNFAPT(I2+1)-1
                DO 250 I4=IBEG4,IEND4
                        IHEAD2=DNFADJ(I4)
                        IHPUT4=DNBAPT(IHEAD2)+DNLIST(IHEAD2)
                        IF(IHPUT4.LE.0)GOTO1900
                        DNBADJ(IHPUT4)=I2
                        DNBTOF(IHPUT4)=I4
250             DNLIST(IHEAD2)=DNLIST(IHEAD2)+1
260     CONTINUE
C
C***********************************************************************
C
        DNFVA=0
        DNAUG=0
        DNSTGE=1
C
C STAGE LOOP:
300     CONTINUE
C   FORM NEW LAYERED NETWORK W.R.T. CURRENT FEASIBLE FLOW.
                CALL DNBFS(LAYMAX)
                IF(LAYMAX.GT.0)GOTO1500
C   COMPUTE A MAXIMAL FLOW IN THIS LAYERED NET:
                CALL DNDFS
C   UPDATE FLOW VALUE:
                DNFVA=DNFVA+DNLFVA
                DNAUG=DNAUG+DNLAUG
                DNSTGE=DNSTGE+1
C   ERASE MARKINGS THAT DEFINE LAYERED NETWORK:
                CALL DNCLEA
        GOTO300
C***********************************************************************
C
C CURRENT FLOW IS MAXIMUM:
1500    MAXFLO=DNFVA
        NUMAUG=DNAUG
        NUMSTG=DNSTGE
        CALL DNCLEA
C MARK EXECUTION TIME:
C       CALL GETIME(TIMEND)
C        TIMEND=gettime(it)
        DNELTM=TIMEND-TIMBEG
        ELTIM=DNELTM
C
C MARK THE MIN CUT-SET IN DNFAPT(.) AND DNFADJ(.):
        CALL DNCUT(NNCUT,NACUT)
C
        RETURN
C
1900    IRETN=1
        RETURN
        END
        SUBROUTINE DNBFS(LAYMAX)
C
C
C DIMENSION THESE TO AT LEAST MAX NUMBER OF NODES + 2:
        INTEGER   DNLIST(65538),DNDIST(65538)
        INTEGER   DNFAPT(65538),DNPTRF(65538),DNBAPT(65538),
     *              DNFLAB(65538),DNPTRB(65538)
C DIMENSION THESE TO AT LEAST MAX NUMBER OF ARCS + 1:
      INTEGER DNFADJ(524273),DNFROM(524273),DNTO(524273),DNBADJ(524273)
      INTEGER DNCAP(524273),DNGCAP(524273),DNFLOW(524273),DNBTOF(524273)
C
        COMMON /DN01/DNFAPT/DN02/DNFADJ/DN03/DNCAP/DN04/DNFROM
        COMMON /DN05/DNFLOW/DN06/DNGCAP/DN08/DNPTRF/DN09/DNLIST
        COMMON /DN10/DNBAPT/DN11/DNFLAB/DN12/DNPTRB
C
        INTEGER  DNNODE,NODP12,DNSRCE,DNSINK,I2,II2,K2,J2,IHEAD2,
     *  ITAIL2,QHEAD2,QTAIL2,KP2,KSAT2,DNNOP2,LAYMAX,K2DST,K2DP1,MAXD2
        INTEGER  DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,DNIBIG
        COMMON /DN00/DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,
     *               DNELTM,DNIBIG,DNNODE,DNSRCE,DNSINK,DNNOP2
        EQUIVALENCE (DNFLOW(1),DNTO(1)),(DNFLAB(1),DNDIST(1)),
     *              (DNFROM(1),DNBADJ(1)),(DNGCAP(1),DNBTOF(1))
C
C*******************************************************************
C
C CALLING CONDITIONS:
C       INTERNAL CALL FROM DNSUB ONLY.
C INPUT ARRAYS:
C       DNFAPT, DNBAPT, DNFADJ, DNBADJ, DNBTOF, DNCAP, DNFLOW
C SCRATCH ARRAYS:
C       DNLIST
C OUTPUT:
C   IN CALLING SEQUENCE (INTEGER*2):
C       LAYMAX: 0 (LAYERED NET COMPLETE WITH SINK IN LAST LAYER.)
C               1 (LAST LAYER IS EMPTY; CURRENT FLOW IN ORIGINAL
C                  NETWORK IS MAXIMUM.)
C   IN COMMON (ARRAYS):
C       DNPTRF, DNPTRB, DNDIST
C***********************************************************************
C
C   THIS SUBROUTINE CONSTRUCTS THE LAYERED NETWORK OF THE RESIDUAL
C   NETWORK W.R.T. THE CURRENT FLOW.  THE RESIDUAL NETWORK IS
C   INFERRED, AND THE LAYERED NETWORK IS ONLY RECORDED BY TEMPORARY
C   MARKERS ON THE ORIGINAL GRAPH. THE SUBROUTINE IMPLEMENTS A
C   BREADTH-FIRST SEARCH ON THE RESIDUAL NETWORK, I.E.  NODES ARE
C   PROCESSED AS IN A QUEUE.  SUCH A SEARCH PARTITIONS THE NODES INTO
C   A SET OF 'LAYERS', ACCORDING TO THEIR CARDNALITY DISTANCE FROM
C   THE source. THE DNDIST(.) ARRAY IS USED TO RECORD THESE DISTANCES,
C   AND THUS DEFINES THE LAYERS (NOTE: DNDIST(ISOURC)=0).
C
C   IN SELECTING ARCS OF THE RESIDUAL NETWORK THAT ARE DIRECTED FROM
C   ONE LAYER TO THE NEXT, WE SHALL MARK THE ARCS IN THE ORIGINAL
C   NETWORK AS 'OPEN' OR 'CLOSED'.  A 'CLOSED' ARC (V,W) IN THE
C   LAYERED NETWORK IS EITHER AN ARC I=(V,W) THAT IS SATURATED IN THE
C   ORIGINAL NETWORK, OR AN ARC I=(W,V) OF THE ORIGINAL NETWORK WITH
C   ZERO FLOW. IN THE FORMER CASE, THE ABSENCE OF THIS ARC FROM THE
C   LAYERED NETWORK IS MARKED BY NEGATING DNFADJ(I), AND IN THE LATTER
C   CASE, BY NEGATING DNBADJ(I).  AN 'OPEN' ARC I=(V,W) FOR THE LAYERED
C   NETWORK EITHER HAS POSITIVE RESIDUAL CAPACITY, OR ARC (W,V) HAS
C   POSITIVE FLOW, IN THE ORIGINAL NETWORK. IF A NODE V HAS NO OPEN
C   ARCS IN THE LAYERED NETWORK THAT ARE IN THE FORWARD ADJACENCY LIST
C   OF V IN THE ORIGINAL NETWORK, THEN WE SET DNPTRF(V)=0; AND IF IT HAS
C   NO OPEN ARCS THAT ARE IN THE BACKWARD ADJACENCY LIST OF V IN THE
C   ORIGINAL NETWORK, WE SET DNPTRB(V)=0.
C
C   IN THIS BFS SEARCH WE ONLY USE OPEN ARCS THAT LEAD US TO NEW
C   NODES, OR TO NODES THAT HAVE ALREADY BEEN PLACED IN THE NEXT
C   LAYER. INITIALLY, THE QUEUE CONTAINS ONLY THE source.  AS THE
C   VERTICES ARE POPPED FROM THE QUEUE AND SCANNED, NEW NODES ARE
C   INJECTED INTO THE QUEUE. EVENTUALLY, EITHER THE SINK IS REACHED,
C   OR SOME VERTICES ARE NOT REACHABLE WITH OPEN ARCS FROM THE
C   CURRENT LAYER. IN THE FORMER CASE THE LAYERED NETWORK FOR THE
C   CURRENT STAGE IS COMPLETE, AND THUS A FLOW AUGMENTATION IS
C   POSSIBLE.  IN THE LATTER CASE, THE CURRENT FLOW ON THE ORIGINAL
C   NETWORK IS MAXIMUM, AND THE RUN TERMINATES.
C
C   THE QUEUE IS MAINTAINED IN THE LIST DNLIST(.) WITH TWO POINTERS,
C   AS SHOWN:
C                   -------------------------------------------
C    ARRAY              |    NODES TO BE SCANNED    |
C    DNLIST(.):  ...    |                           |
C                   -------------------------------------------
C                      ^                           ^
C                      |                           |
C                    QHEAD2                     QTAIL2
C*******************************************************************
C
C INITIALIZE:
                DO 10 I2=1,DNNODE
                        DNPTRF(I2)=0
                        DNPTRB(I2)=0
10              DNDIST(I2)=DNNOP2
                DNDIST(DNSRCE)=0
C PUT source INTO QUEUE:
                QHEAD2=0
                QTAIL2=1
                DNLIST(1)=DNSRCE
                MAXD2=DNNODE-1
C
C SCAN EACH NODE IN QUEUE:
C
C---------------
C
100     IF(QHEAD2.EQ.QTAIL2)GOTO2000
                QHEAD2=QHEAD2+1
C POP NODE IN FRONT OF QUEUE:
                K2=DNLIST(QHEAD2)
                K2DST=DNDIST(K2)
        IF(K2DST.GE.MAXD2)GOTO100
                K2DP1=K2DST+1
C
C  SCAN NODE K2, I.E. SEARCH OVER FWD ADJACENCY OF K2 AND FOR ARCS
C  (K2,J2) SUCH THAT J2 IS UNSCANNED AND J2 IS NOT IN THE QUEUE AND
C  ARC (K2,J2) HAS POSITIVE RESIDUAL CAPACITY IN ORIGINAL NETWORK.
C  'J2 UNSCANNED' IS CHECKED BY THE CONDITION: 'DNDIST(J2)>=DNDIST(K2)'.
C  'J2 NOT IN QUEUE' IS CHECKED BY THE CONDITION
C  'DNDIST(J2)=DNDIST(K2)+1 FOR      A SCANNED J2'.
C
                IBEG4=DNFAPT(K2)
                IEND4=DNFAPT(K2+1)-1
                IF(IBEG4.GT.IEND4)GOTO1000
                DO 400 I4=IBEG4,IEND4
                        J2=DNFADJ(I4)
                        IF(DNDIST(J2).LE.K2DST)GOTO300
                        IF(DNCAP(I4).LE.DNFLOW(I4))GOTO300
C (IMPLICITLY) MARK ARC (K2,J2) AS OPEN IN LAYERED NETWORK; ALSO,
C SAVE ITS INDEX IN DNPTRF(.):
                        DNPTRF(K2)=I4
                        IF(J2.NE.DNSINK)GOTO200
C J2=SINK:
                        MAXD2=K2DP1
                        DNDIST(J2)=MAXD2
                        GOTO400
C APPEND NODE J2 TO THE QUEUE, IF NOT ALREADY THERE:
200                     IF(DNDIST(J2).EQ.K2DP1)GOTO400
                        DNDIST(J2)=K2DP1
                        QTAIL2=QTAIL2+1
                        DNLIST(QTAIL2)=J2
                        GOTO400
C (EXPLICITLY) MARK ARC (K2,J2) AS CLOSED IN LAYERED NETWORK:
300                     DNFADJ(I4)=-DNFADJ(I4)
400             CONTINUE
C
C---------------
C
C IF SINK WAS REACHED, DON'T NEED TO SCAN BACKWARD ARCS      INTO ITS LAYER:
1000      IF(DNDIST(DNSINK).NE.DNNOP2)GOTO100
C
C ELSE,      CONTINUE SCAN OVER FORWARD ARCSOUT OF K2 IN THE RESIDUAL
C NETWORK (WHICH ARE BACKWARD ARCS OF THE ORIGINAL NETWORK WITH
C POSITIVE FLOW). THIS CODE SEGMENT IS ANALOGOUS TO THE      ONE ABOVE:
C
      IBEG4=DNBAPT(K2)
      IEND4=DNBAPT(K2+1)-1
      IF(IBEG4.GT.IEND4)GOTO100
      DO 1400 I4=IBEG4,IEND4
        J2=DNBADJ(I4)
        IF(DNDIST(J2).LE.K2DST)GOTO1300
        JJDN=DNBTOF(I4)
        IF(DNFLOW(JJDN).LE.0)GOTO1300
        DNPTRB(K2)=I4
        IF(DNDIST(J2).EQ.K2DP1)GOTO1400
        DNDIST(J2)=K2DP1
        QTAIL2=QTAIL2+1
        DNLIST(QTAIL2)=J2
        GOTO1400
 1300       DNBADJ(I4)=-DNBADJ(I4)
 1400      CONTINUE
      GOTO100
C
C---------------
C
2000      LAYER2=LAYER2+1
      LAYP12=LAYER2+1
C HERE,      ALL NODES IN QUEUE HAVEBEEN PROCESSED.
C IF THE SINK WAS REACHED, THE LAYERED NETWORK IS COMPLETE:
      LAYMAX=0
      IF(DNDIST(DNSINK).NE.DNNOP2)RETURN
C ELSE,      THE CURRENT FLOW IS MAXIMUM:
      LAYMAX=1
      RETURN
C
      END
      SUBROUTINE DNDFS
C
C
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF NODES + 2:
      INTEGER  DNLIST(65538),DNDIST(65538)
      INTEGER  DNFAPT(65538),DNPTRF(65538),DNBAPT(65538),
     *          DNFLAB(65538),DNPTRB(65538)
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF ARCS + 1:
      INTEGER DNFADJ(524273),DNFROM(524273),DNTO(524273),DNBADJ(524273)
      INTEGER DNCAP(524273),DNGCAP(524273),DNFLOW(524273),DNBTOF(524273)
C
      COMMON /DN01/DNFAPT/DN02/DNFADJ/DN03/DNCAP/DN04/DNFROM
      COMMON /DN05/DNFLOW/DN06/DNGCAP/DN08/DNPTRF/DN09/DNLIST
      COMMON /DN10/DNBAPT/DN11/DNFLAB/DN12/DNPTRB
C
      INTEGER DNNODE,NODP12,DNSRCE,DNSINK,I2,II2,K2,J2,IHEAD2,
     *      ITAIL2,QHEAD2,QTAIL2,KP2,KSAT2,DNNOP2,LAYMAX,K2DST,K2DP1,MAXD2
      INTEGER DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,DNIBIG
      COMMON /DN00/DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,
     *           DNELTM,DNIBIG,DNNODE,DNSRCE,DNSINK,DNNOP2
      EQUIVALENCE (DNFLOW(1),DNTO(1)),(DNFLAB(1),DNDIST(1)),
     *          (DNFROM(1),DNBADJ(1)),(DNGCAP(1),DNBTOF(1))
C
C***********************************************************************
C
C  THIS      SUBROUTINE FINDS A MAXIMAL FLOWIN THE LAYERED NETWORK.	IT
C  USES      A DEPTH-FIRST SEARCH STARTING FROM THE SOURCE AND RECORDS
C  THE DFS TREE      BY THE PREDECESSOR ARRAY DNLIST(.) AND FLOW LABEL
C  ARRAY DNFLAB(.). LAYERED NETWORK ARCS THAT ARE FORWARD IN THE
C  ORIGINAL NETWORK ARE      SEARCHED BEFOREBACKWARD ARCS.	THE
C  FORWARD AND BACKWARD      ADJACENCIES OF A NODE K2 ARE SCANNED INLIFO
C  ORDER, STARTING FROM      THE LAST ARC THAT WAS VISITED DURING A
C  PREVIOUS SCAN OF K2.      THUS, WITH THE AID OF THE TWO POINTER LISTS
C  DNPTRF(.) AND DNPTRB(.), THE      ADJACENCY LISTSOF THE ORIGINAL
C  NETWORK ARE SCANNED ONLY ONCE BY THIS SUBROUTINE. DNPTRF(K2)=0
C  IMPLIES THAT      NODE K2HAS NO OUTGOING	UNSCANNED ARCS;	DNPTRB(K2)=0
C  IMPLIES THAT      NODE K2HAS NO INCOMING	UNSCANNED ARCS;	A 'CLOSED'
C  NODE      IS DETECTED BY THE CONDITION DNPTRF(K2)=DNPTRB(K2)=0.
C
C***********************************************************************
C
C CALLING CONDITIONS:
C      INTERNAL CALL FROM DNSUB ONLY.
C INPUT      ARRAYS:
C      DNPTRB,DNPTR, DNBTOF, DNCAP, DNFLOW, DNBADJ, DNFADJ
C OUTPUT ARRAYS:
C      DNPTRF,DNPTRB,	DNLIST,	DNFLAB.
C***********************************************************************
C
C      SUBROUTINES CALLED:DNPUSH
C
C***********************************************************************
C
      K2=DNSRCE
      DNLFVA=0
      DNLAUG=0
      DNFLAB(K2)=DNIBIG
C
C-----------
C
C SCAN NODE K2:
C
100      IF(DNPTRF(K2).EQ.0)GOTO1000
C FIND AN OPEN ARC FROM      K2 TO SOME NODEJ2: SCAN THE
C FORWARD ADJACENCY LIST OF K2,      STARTING FROM  DNPTRF(K2) AND
C PROCEEDING BACKWARD ON DNFADJ() TOWARD DNFAPT(K2):
      I4=DNPTRF(K2)
      IPT4=DNFAPT(K2)
300      J2=DNFADJ(I4)
        IF(J2.GE.0)GOTO600
  400        I4=I4-1
      IF(I4.GE.IPT4)GOTO300
C NO OPEN ARCS OUT OF K2; MARK K2 AS CLOSED:
      DNPTRF(K2)=0
      GOTO1000
C AN OPEN ARC FOUND; MOVE POINTER DNPTRF() TO THIS ARC;
C EXTEND DFS TREE TO NODE J2:
  600      DNPTRF(K2)=I4
      DNFLAB(J2)=DNFLAB(K2)
      IF(DNCAP(I4)-DNFLOW(I4).LT.DNFLAB(J2))
     *        DNFLAB(J2)=DNCAP(I4)-DNFLOW(I4)
      DNLIST(J2)=K2
      K2=J2
C IF NODE K2 (FORMERLY J2) IS THE SINK,      AUGMENTFLOW; GET NEW K2:
      IF(K2.EQ.DNSINK)CALL DNPUSH(K2)
      GOTO100
C
C-----------
C
 1000      IF(DNPTRB(K2).EQ.0)GOTO2000
C SCAN BACKWARD      ARCS INTO K2 TOFIND AN	OPEN ARC:
      I4=DNPTRB(K2)
      IPT4=DNBAPT(K2)
 1300      J2=DNBADJ(I4)
       IF(J2.GE.0)GOTO1600
       I4=I4-1
       IF(I4.GE.IPT4)GOTO1300
C NO OPEN ARC INTO K2:
      DNPTRB(K2)=0
      GOTO2000
C AN OPEN ARC FOUND; EXTEND DFS      TREE TONODE J2:
 1600      DNPTRB(K2)=I4
      DNFLAB(J2)=DNFLAB(K2)
      JJDN=DNBTOF(I4)
      IFL4=DNFLOW(JJDN)
      IF(IFL4.LT.DNFLAB(J2))DNFLAB(J2)=IFL4
      DNLIST(J2)=-K2
      K2=J2
      GOTO100
C
C-----------
C
C K2 IS      'CLOSED' NODE; IF IT ISTHE source, THEN WE HAVE A MAXIMAL
C FLOW IN LAYERED NETWORK:
2000      IF(K2.EQ.DNSRCE)RETURN
C ELSE,      BACK UPONE NODE FROM K2 IN DFS	TREE:
      KP2=DNLIST(K2)
      K2=KP2
      IF(K2.LT.0)K2=-K2
      IF(KP2.GE.0)GOTO2200
      I4=DNPTRB(K2)
      DNBADJ(I4)=-DNBADJ(I4)
      GOTO100
2200      I4=DNPTRF(K2)
      DNFADJ(I4)=-DNFADJ(I4)
      GOTO100
C
C-----------
      END
      SUBROUTINE DNPUSH(K2)
C
C
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF NODES + 2:
      INTEGER  DNLIST(65538),DNDIST(65538)
      INTEGER  DNFAPT(65538),DNPTRF(65538),DNBAPT(65538),
     *          DNFLAB(65538),DNPTRB(65538)
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF ARCS + 1:
      INTEGER DNFADJ(524273),DNFROM(524273),DNTO(524273),DNBADJ(524273)
      INTEGER DNCAP(524273),DNGCAP(524273),DNFLOW(524273),DNBTOF(524273)
C
      COMMON /DN01/DNFAPT/DN02/DNFADJ/DN03/DNCAP/DN04/DNFROM
      COMMON /DN05/DNFLOW/DN06/DNGCAP/DN08/DNPTRF/DN09/DNLIST
      COMMON /DN10/DNBAPT/DN11/DNFLAB/DN12/DNPTRB
C
      INTEGER DNNODE,NODP12,DNSRCE,DNSINK,I2,II2,K2,J2,IHEAD2,
     *      ITAIL2,QHEAD2,QTAIL2,KP2,KSAT2,DNNOP2,LAYMAX,K2DST,K2DP1,MAXD2
      INTEGER DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,DNIBIG
      COMMON /DN00/DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,
     *           DNELTM,DNIBIG,DNNODE,DNSRCE,DNSINK,DNNOP2
      EQUIVALENCE (DNFLOW(1),DNTO(1)),(DNFLAB(1),DNDIST(1)),
     *          (DNFROM(1),DNBADJ(1)),(DNGCAP(1),DNBTOF(1))
C
C***********************************************************************
C
C AUGMENT FLOW ALONG FLOW AUGMENTING PATH DEFINED SUBROUTINE DNDFS, I.E.
C USING      THE PREDECESSORARRAY DNLIST(.), START FROM THE	SINK AND
C TRAVERSE TO THE SOURCE ARC FLOWS AND FLOW LABELS BY AN AMOUNT      EQUAL
C TO DNFLAB(DNSINK). MARK SATURATED FORWARD (FROM S TO T) ARCS,      AND
C BACKWARD ARCS      HAVING ZERO FLOW, AS CLOSED.
C
C***********************************************************************
C
C CALLING CONDITIONS:
C      INTERNAL CALL FROM DNDFS ONLY.
C INPUT      ARRAYS:
C      DNFLAB,DNLIST,DNPTRB,DNPTRF,DNBTOF,DNCAP,DNFLOW,DNBADJ,DNFADJ
C OUTPUT ARRAYS:
C      DNFLAB,DNFLOW,DNBADJ,DNFADJ
C***********************************************************************
C
      KSAT2=0
      J2=DNSINK
      INCRE4=DNFLAB(J2)
      DNLFVA=DNLFVA+INCRE4
      DNLAUG=DNLAUG+1
C
100      K2=J2
      J2=DNLIST(K2)
      DNFLAB(K2)=DNFLAB(K2)-INCRE4
      IF(J2.GT.0)GOTO200
C
C DECREASE FLOW      ON BACKWARD ARC(K2,J2):
      J2=-J2
      I4=DNPTRB(J2)
      II4=DNBTOF(I4)
      DNFLOW(II4)=DNFLOW(II4)-INCRE4
      IF(DNFLOW(II4).NE.0)GOTO100
C FLOW IS ZERO;      MARK ARC AS 'CLOSED' INLAYERED	NETWORK:
      DNBADJ(I4)=-DNBADJ(I4)
      KSAT2=J2
      GOTO100
C
C INCREASE FLOW      ON FORWARD ARC (J2,K2):
200      I4=DNPTRF(J2)
      DNFLOW(I4)=DNFLOW(I4)+INCRE4
      IF(DNCAP(I4).NE.DNFLOW(I4))GOTO300
C ARC IS NOW SATURATED;      MARK ISAS 'CLOSED' IN LAYERED NET
      DNFADJ(I4)=-DNFADJ(I4)
      KSAT2=J2
300      IF(J2.NE.DNSRCE)GOTO100
C
C RETURN TO RESUME SEARCH FROM K2 (NODE      CLOSESTTO source OF
C THE ARC CLOSED LAST)
      IF(KSAT2.EQ.0)STOP
      K2=KSAT2
      RETURN
      END
      SUBROUTINE DNFWD(NODES,NARCS,IRETN)
C
C
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF NODES + 2:
      INTEGER  DNLIST(65538),DNDIST(65538)
      INTEGER  DNFAPT(65538),DNPTRF(65538),DNBAPT(65538),
     *          DNFLAB(65538),DNPTRB(65538)
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF ARCS + 1:
      INTEGER DNFADJ(524273),DNFROM(524273),DNTO(524273),DNBADJ(524273)
      INTEGER DNCAP(524273),DNGCAP(524273),DNFLOW(524273),DNBTOF(524273)
C
      COMMON /DN01/DNFAPT/DN02/DNFADJ/DN03/DNCAP/DN04/DNFROM
      COMMON /DN05/DNFLOW/DN06/DNGCAP/DN08/DNPTRF/DN09/DNLIST
      COMMON /DN10/DNBAPT/DN11/DNFLAB/DN12/DNPTRB
C
      INTEGER DNNODE,NODP12,DNSRCE,DNSINK,I2,II2,K2,J2,IHEAD2,
     *      ITAIL2,QHEAD2,QTAIL2,KP2,KSAT2,DNNOP2,LAYMAX,K2DST,K2DP1,MAXD2
      INTEGER DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,DNIBIG
      COMMON /DN00/DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,
     *           DNELTM,DNIBIG,DNNODE,DNSRCE,DNSINK,DNNOP2
      EQUIVALENCE (DNFLOW(1),DNTO(1)),(DNFLAB(1),DNDIST(1)),
     *          (DNFROM(1),DNBADJ(1)),(DNGCAP(1),DNBTOF(1))
C
C***********************************************************************
C
C CALLING CONDITIONS:
C      USER CALLABLE; BEFORE CALLING DNSUB.
C INPUT:
C   IN CALLING SEQUENCE      (ALL INTEGER*4):
C      NODES=NUMBER OF NODES, INCLUDING source AND SINK
C      NARCS=NUMBER OF ARCS.
C
C   IN COMMON, PROVIDE THREE ARC LISTS WITH ARCS IN ANY      ORDER:
C      DNFROM=THE LIST OF TAILS. (INTEGER*2)
C      DNTO=THE LIST OF HEADS. (INTEGER*2)
C      DNGCAP=THE LIST OF CAPACITIES.	(INTEGER*4).
C
C SCRATCH ARRAY      (IN COMMON):
C      DNLIST
C OUTPUT:
C   IN CALLING SEQUENCE      (INTEGER*4):
C      IRETN=0 NO ERRORS;  =1	ERROR DETECTED,	CHECK INPUT.
C   IN COMMON, ARRAYS:
C      DNFAPT,DNFADJ,DNCAP (I.E. THE FWD ADJACENCY INPUT DATA STRUCTURE
C       THAT IS REQUIRED AS INPUT TO SUBROUTINE DNSUB).
C
C***********************************************************************
C
C  USING THE THREE ARC LISTS DNFROM(.),      DNTO(.), DNGCAP(.), THIS
C  SUBROUTINE CONSTRUCTS THE FORWARD ADJACENCY ARRAYS DNFAPT(.),
C  DNFADJ(.), AND DNCAP(.), AS THE INPUT REQUIRED BY SUBROUTINE      DNSUB.
C
C***********************************************************************
C
C INITIALIZE:
      IRETN=0
      NODP1=NODES+1
      DO 100 I2=1,NODP1
      DNFAPT(I2)=0
100      DNLIST(I2)=0
C TEMPORARILY STORE IN DNFAPT(.) NUMBER      OF ARCSOUT OF EACH NODE:
      DO 200 I4=1,NARCS
        JJDN=DNFROM(I4)
200      DNFAPT(JJDN)=DNFAPT(JJDN)+1
C CONSTRUCT DNFAPT(.):
      ITPI4=DNFAPT(1)
      DNFAPT(1)=1
      DO 300 I2=1,NODES
      ITPSV4=ITPI4+DNFAPT(I2)
      ITPI4=DNFAPT(I2+1)
300      DNFAPT(I2+1)=ITPSV4
C CONSTRUCT DNFADJ(.) AND DNCAP(.):
      DO 400 I4=1,NARCS
      ITAIL2=DNFROM(I4)
      ITPUT4=DNFAPT(ITAIL2)+DNLIST(ITAIL2)
      IF(ITPUT4.LE.0)GOTO500
      DNFADJ(ITPUT4)=DNTO(I4)
      DNLIST(ITAIL2)=DNLIST(ITAIL2)+1
      DNCAP(ITPUT4)=DNGCAP(I4)
400      CONTINUE
      RETURN
500      IRETN=1
      RETURN
      END
      SUBROUTINE DNCLEA
C
C
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF NODES + 2:
      INTEGER  DNLIST(65538),DNDIST(65538)
      INTEGER  DNFAPT(65538),DNPTRF(65538),DNBAPT(65538),
     *          DNFLAB(65538),DNPTRB(65538)
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF ARCS + 1:
      INTEGER DNFADJ(524273),DNFROM(524273),DNTO(524273),DNBADJ(524273)
      INTEGER DNCAP(524273),DNGCAP(524273),DNFLOW(524273),DNBTOF(524273)
C
      COMMON /DN01/DNFAPT/DN02/DNFADJ/DN03/DNCAP/DN04/DNFROM
      COMMON /DN05/DNFLOW/DN06/DNGCAP/DN08/DNPTRF/DN09/DNLIST
      COMMON /DN10/DNBAPT/DN11/DNFLAB/DN12/DNPTRB
C
      INTEGER DNNODE,NODP12,DNSRCE,DNSINK,I2,II2,K2,J2,IHEAD2,
     *      ITAIL2,QHEAD2,QTAIL2,KP2,KSAT2,DNNOP2,LAYMAX,K2DST,K2DP1,MAXD2
      INTEGER DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,DNIBIG
      COMMON /DN00/DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,
     *           DNELTM,DNIBIG,DNNODE,DNSRCE,DNSINK,DNNOP2
      EQUIVALENCE (DNFLOW(1),DNTO(1)),(DNFLAB(1),DNDIST(1)),
     *          (DNFROM(1),DNBADJ(1)),(DNGCAP(1),DNBTOF(1))
C
C*******************************************************************
C CALLING CONDITIONS:
C      INTERNAL CALL FROM DNSUB ONLY.
C INPUT      (IN COMMON):
C   SCALARS: DNARC
C   ARRAYS:  DNFADJ, DNBADJ
C OUTPUT:
C   ARRAYS (IN COMMON):      DNFADJ,DNBADJ
C********************************************************************
C
C CLEAR      MARKINGS IN ARRAYS DNFADJ() ANDDNBADJ().:
      DO 100 I4=1,DNARC
        IF(DNBADJ(I4).LT.0)DNBADJ(I4)=-DNBADJ(I4)
        IF(DNFADJ(I4).LT.0)DNFADJ(I4)=-DNFADJ(I4)
100      CONTINUE
      RETURN
      END
      SUBROUTINE DNCUT(NNCUT,NACUT)
C
C
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF NODES + 2:
      INTEGER  DNLIST(65538),DNDIST(65538)
      INTEGER  DNFAPT(65538),DNPTRF(65538),DNBAPT(65538),
     *          DNFLAB(65538),DNPTRB(65538)
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF ARCS + 1:
      INTEGER DNFADJ(524273),DNFROM(524273),DNTO(524273),DNBADJ(524273)
      INTEGER DNCAP(524273),DNGCAP(524273),DNFLOW(524273),DNBTOF(524273)
C
      COMMON /DN01/DNFAPT/DN02/DNFADJ/DN03/DNCAP/DN04/DNFROM
      COMMON /DN05/DNFLOW/DN06/DNGCAP/DN08/DNPTRF/DN09/DNLIST
      COMMON /DN10/DNBAPT/DN11/DNFLAB/DN12/DNPTRB
C
      INTEGER DNNODE,NODP12,DNSRCE,DNSINK,I2,II2,K2,J2,IHEAD2,
     *      ITAIL2,QHEAD2,QTAIL2,KP2,KSAT2,DNNOP2,LAYMAX,K2DST,K2DP1,MAXD2
      INTEGER DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,DNIBIG
      COMMON /DN00/DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,
     *           DNELTM,DNIBIG,DNNODE,DNSRCE,DNSINK,DNNOP2
      EQUIVALENCE (DNFLOW(1),DNTO(1)),(DNFLAB(1),DNDIST(1)),
     *          (DNFROM(1),DNBADJ(1)),(DNGCAP(1),DNBTOF(1))
C
C***********************************************************************
C CALLING CONDITIONS:
C      ONLY AFTER A SUCCESSFULRETURN FROM A CALL TO DNSUB.
C INPUT      ARRAYS:
C      DNFAPT,DNFADJ,	DNCAP,DNFLOW, DNDIST
C OUTPUT ARRAYS:
C      DNFAPT,DNFADJ
C PASSED IN CALLING SEQUENCE:
C      NNCUT:NUMBER OF NODES	ON source SIDE OF FINAL	CUT (INTEGER*4)
C      NACUT:NUMBER OF ARCS IN THE FINAL CUT	(INTEGER*4)
C***********************************************************************
C
C NEGATE DNFAPT(.) FOR NODES ON      SINK SIDE OF CUT:
      NNCUT=0
      DO 1700I2=1,DNNODE
      IF(DNDIST(I2).NE.DNNOP2)GOTO1650
      DNFAPT(I2)=-IABS(DNFAPT(I2))
      GOTO1700
1650      NNCUT=NNCUT+1
1700      CONTINUE
C NEGATE DNFADJ(.) FOR THOSE SATURATED ARCS IN MIN CUT:
      NACUT=0
      DO 1900I2=1,DNNODE
      IF(DNFAPT(I2).LT.0)GOTO1900
C NODE I2 IS ON      source SIDE:
      IBEG4=IABS(DNFAPT(I2))
      IEND4=IABS(DNFAPT(I2+1))-1
      DO 1800 I4=IBEG4,IEND4
        JJDN=DNFADJ(I4)
        IF(DNFAPT(JJDN).GT.0)GOTO1800
C NODE DNFADJ(I4) ON SINK SIDE:
        IF(DNFLOW(I4).NE.DNCAP(I4))GOTO1800
        II2=DNFADJ(I4)
        DNFADJ(I4)=-IABS(II2)
        NACUT=NACUT+1
 1800      CONTINUE
 1900      CONTINUE
      RETURN
      END
      SUBROUTINE DNOUT(IUNREP)
C
        CHARACTER*1  T  , JJS,  JJT , ISIDE, JSIDE 
        CHARACTER*3  BL3, BSAT, TAG1
        CHARACTER*4  BL4, BCUT, TAG2
C
C
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF NODES + 2:
      INTEGER  DNLIST(65538),DNDIST(65538)
      INTEGER  DNFAPT(65538),DNPTRF(65538),DNBAPT(65538),
     *          DNFLAB(65538),DNPTRB(65538)
C DIMENSION THESE TO AT      LEAST MAX NUMBER OF ARCS + 1:
      INTEGER DNFADJ(524273),DNFROM(524273),DNTO(524273),DNBADJ(524273)
      INTEGER DNCAP(524273),DNGCAP(524273),DNFLOW(524273),DNBTOF(524273)
C
      COMMON /DN01/DNFAPT/DN02/DNFADJ/DN03/DNCAP/DN04/DNFROM
      COMMON /DN05/DNFLOW/DN06/DNGCAP/DN08/DNPTRF/DN09/DNLIST
      COMMON /DN10/DNBAPT/DN11/DNFLAB/DN12/DNPTRB
C
      INTEGER DNNODE,NODP12,DNSRCE,DNSINK,I2,II2,K2,J2,IHEAD2,
     *      ITAIL2,QHEAD2,QTAIL2,KP2,KSAT2,DNNOP2,LAYMAX,K2DST,K2DP1,MAXD2
      INTEGER DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,DNIBIG
      COMMON /DN00/DNARC,DNFVA,DNAUG,DNLFVA,DNLAUG,DNSTGE,
     *           DNELTM,DNIBIG,DNNODE,DNSRCE,DNSINK,DNNOP2
      EQUIVALENCE (DNFLOW(1),DNTO(1)),(DNFLAB(1),DNDIST(1)),
     *          (DNFROM(1),DNBADJ(1)),(DNGCAP(1),DNBTOF(1))
        DATA BL3/3H   /,  BL4/4H    /,  BSAT/3HSAT/,  BCUT/4H-CUT/,
     1       JJT/1HT/
      DATA JJS/1HS/
C
C***********************************************************************
C
C PREPARES OUTPUT REPORT. MAY ONLY BE CALLED AFTER A CALL TO DNSUB.
C
C***********************************************************************
C
C CALLING CONDITIONS:
C      USER CALLABLE; ONLY AFTER CALLING DNSUB.
C INPUT      ARRAYS:
C      DNFAPT,DNFADJ,	DNCAP, DNFLOW
C SCRATCH ARRAYS:
C      DNLIST
C OUTPUT ARRAYS:  NONE
C***********************************************************************
C
C PRINT      SUMMARYSECTION:
      WRITE(IUNREP,1000)
     *            DNNODE,DNARC,DNSRCE,DNSINK,DNFVA,DNELTM,DNAUG,DNSTGE
C PRINT      CUT-SETNODES:
      K2=0
      DO 200 I2=1,DNNODE
      IF(DNFAPT(I2).LT.0)GOTO200
      K2=K2+1
      DNLIST(K2)=I2
200      CONTINUE
c     WRITE(IUNREP,1003)K2
c     WRITE(IUNREP,1004)(DNLIST(I2),I2=1,K2)
c     WRITE(IUNREP,1001)
C PRINT      ARC LIST:
      NTOTC4=0
      NSAINC=0
      NSA=0
      NPOSFL=0
      DO 400 I2=1,DNNODE
         IBEG4=IABS(DNFAPT(I2))
         IEND4=IABS(DNFAPT(I2+1))-1
         DO 300 I4=IBEG4,IEND4
      MFLO4=DNFLOW(I4)
      IF(MFLO4.EQ.0)GOTO300
      NPOSFL=NPOSFL+1
      TAG1=BL3
      IF(MFLO4.NE.DNCAP(I4))GOTO270
        TAG1=BSAT
        NSA=NSA+1
 270      TAG2=BL4
C      IF(DNFADJ(I4).LT.0)TAG2=BCUT
      IF(DNFADJ(I4).GE.0)GOTO280
        NTOTC4=NTOTC4+MFLO4
        NSAINC=NSAINC+1
  280      ISIDE=JJS
      JSIDE=ISIDE
      IF(DNFAPT(I2).LT.0)ISIDE=JJT
      IFAPLU=DNFADJ(I4)
      IFAPLU=IABS(IFAPLU)
      IF(DNFAPT(IFAPLU).LT.0)JSIDE=JJT
      IF(ISIDE.NE.JSIDE)TAG2=BCUT
C
c     WRITE(IUNREP,1002)I2,ISIDE,IFAPLU,JSIDE,
c    *        MFLO4,DNCAP(I4),TAG1,TAG2
  300         CONTINUE
  400      CONTINUE
C PRINT      MAXFLO VALUE AND COMPUTED MINCUT
c     WRITE(IUNREP,1006)NPOSFL,NSA,NSAINC
c     IF(DNFVA.NE.NTOTC4)WRITE(IUNREP,1007)DNFVA,NTOTC4
c     WRITE(IUNREP,1008)
      RETURN
C
C FORMATS:
1000      FORMAT(/13H DINIC MAXFLO,
     */13H PROBLEM HAS ,I6,7H NODES,,I6,24H ARCS, SOURCE/SINK PAIR,,2I6
     */16H MAX FLOW VALUE=,I16,21H, PC/AT ELAPSED TIME ,F8.2,6H SECS.,
     */21H TOTAL AUGMENTATIONS=,I6,4H IN ,I6,7H STAGES)
1001      FORMAT(/25H ARCS WITH POSITIVE FLOW , 
     *//23H       TAIL        HEAD,
     *      40H           FLOW       CAPACITY    STATUS/)
1002      FORMAT(I10,1X,A1,I10,1X,A1,2I15,4X,A3,A4)
1003      FORMAT(/10H THERE ARE,I8,29H NODES IN SOURCE SIDE OF CUT,,
     *      10H THEY ARE /)
1004      FORMAT(10I6)
1006      FORMAT(/7H NOTES ,
     */10X,13H 1) THERE ARE,I6,26H ARCS WITH POSITIVE FLOW, ,I5,3H OF
     */10X,22H      THEM SATURATED (,I5,26H OF WHICH ARE IN MINCUT). ,
     */10X,52H 2)   S  /  T   REFER TO SOURCE / SINK SIDES OF CUT.)
 1007      FORMAT(
     * 10X,41H 3) MAX FLOW VALUE FROM SOURCE TO SINK IS,I14,1H,
     */10X,41H      BUT, CUT CAPACITY WAS COMPUTED AS  ,I14)
1008      FORMAT(/15H END OF REPORT.)
      END


