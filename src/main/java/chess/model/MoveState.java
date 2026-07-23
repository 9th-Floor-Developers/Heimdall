package chess.model;

public record MoveState(Move move,
                        PieceType capturedPieceType,
                        Color capturedColor,
                        boolean whiteCanCastleKingside,
                        boolean whiteCanCastleQueenside,
                        boolean blackCanCastleKingside,
                        boolean blackCanCastleQueenside,
                        int enPassantTarget,
                        int halfMoveClock) {}
